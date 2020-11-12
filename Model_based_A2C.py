# Author: Fu Yangqing
# NUS ID: A0225413R
# Description:
# Use A2C algorithm to train quadrotor to do compliance motion (Discrete action space)
# Update time:2020/10/30


import time
import random as rd
import argparse
import numpy as np
import tensorflow as tf
import csv
import keras

from keras.layers import Conv2D, MaxPooling2D, Dense, GRU, Input, ELU

from keras.optimizers import Adam

from keras.models import Model

from PIL import Image

import os

from Model_Based_ENV import windENV, object_pos




class A2CAgent(object):

    def __init__(self, state_size, action_size, actor_lr, critic_lr,

                 gamma, lambd, entropy, updatetime, load_model):

        self.actor_lr = actor_lr

        self.critic_lr = critic_lr

        self.state_size = state_size

        self.action_size = action_size

        self.acc_size = 3

        self.gamma = gamma

        self.lambd = lambd

        self.entropy = entropy

        self.actor, self.critic = self.build_model()

        _, self.critic2 = self.build_model()

        self.actor_update = self.build_actor_optimizer()

        self.critic_update = self.build_critic_optimizer()

        self.updatetime = updatetime

        if load_model:

            self.load_model('./save_model/' + agent_name)

        self.critic2.set_weights(self.critic.get_weights())

        self.states, self.actions, self.rewards = [], [], []

    def build_model(self):

        # shared network

        x1 = Input(shape=self.state_size)

        layer1 = keras.layers.BatchNormalization()(x1)

        layer2 = keras.layers.TimeDistributed(Conv2D(8, (8, 8), activation='relu', padding='valid'))(layer1)

        layer3 = keras.layers.TimeDistributed(MaxPooling2D((2, 2)))(layer2)

        layer4 = keras.layers.TimeDistributed(Conv2D(16, (5, 5), activation='relu'))(layer3)

        layer5 = keras.layers.TimeDistributed(MaxPooling2D((2, 2)))(layer4)

        layer6 = keras.layers.TimeDistributed(
            Conv2D(32, (3, 3), activation='relu'))(layer5)

        layer7 = keras.layers.TimeDistributed(MaxPooling2D((2, 2)))(layer6)

        layer8 = keras.layers.TimeDistributed(Conv2D(16, (1, 1), activation='relu'))(layer7)

        layer9 = keras.layers.Dropout(rate=0.2)(layer8)

        layer10 = keras.layers.TimeDistributed(keras.layers.Flatten())(layer9)

        layer11 = GRU(32, kernel_initializer='he_normal', use_bias=False)(layer10)

        layer12 = keras.layers.BatchNormalization()(layer11)

        image_out = keras.layers.Activation('sigmoid')(layer12)

        # acc process

        acc = Input(shape=[self.acc_size])

        acc_process = Dense(32)(acc)

        acc_process = keras.layers.BatchNormalization()(acc_process)

        acc_process = keras.layers.Activation('sigmoid')(acc_process)

        state_process = keras.layers.Add()([image_out,acc_process])

        # Actor

        policy = Dense(16, kernel_initializer='he_normal', use_bias=False)(state_process)

        policy = ELU()(policy)

        policy = keras.layers.BatchNormalization()(policy)

        policy = Dense(self.action_size, activation='softmax',
                       kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(policy)

        actor = Model(inputs=[x1, acc], outputs=policy)

        # Critic

        value = Dense(32, kernel_initializer='he_normal', use_bias=False)(state_process)

        value = ELU()(value)

        value = keras.layers.BatchNormalization()(value)

        value = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(value)

        critic = Model(inputs=[x1, acc], outputs=value)

        actor._make_predict_function()

        critic._make_predict_function()

        return actor, critic

    def build_actor_optimizer(self):

        action = keras.backend.placeholder(shape=[None, self.action_size])

        advantages = keras.backend.placeholder(shape=[None, ])

        policy = self.actor.output

        action_prob = keras.backend.sum(action * policy, axis=1)

        log_prob_actions_v = keras.backend.log(action_prob + 1e-6) * advantages

        loss_p = -keras.backend.sum(log_prob_actions_v)

        loss = keras.backend.mean(loss_p)

        optimizer = Adam(lr=self.actor_lr)

        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)

        train = keras.backend.function([self.actor.input[0], self.actor.input[1], action, advantages],

                                       [loss], updates=updates)

        return train

    def build_critic_optimizer(self):

        policy = self.actor.output

        y = keras.backend.placeholder(shape=(None, 1))

        value = self.critic.output

        # # Huber loss

        error = tf.abs(y - value)

        quadratic = keras.backend.clip(error, 0.0, 1.0)

        linear = error - quadratic

        entropy = keras.backend.sum(policy * keras.backend.log(policy + 1e-6), axis=1)

        entropy_loss = -self.entropy*entropy

        value_loss = keras.backend.mean(0.5 * keras.backend.square(quadratic) + linear)

        loss = value_loss + entropy_loss

        optimizer = Adam(lr=self.critic_lr)

        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)

        train = keras.backend.function([self.critic.input[0], self.critic.input[1], y],

                                       [loss], updates=updates)

        return train

    def get_action(self, state):

        policy = self.actor.predict(state)[0]

        policy = np.array(policy)

        action = np.random.choice(self.action_size, 1, p=policy)[0]

        return action, policy

    def train_model(self, next_state, done):

        images = np.zeros([len(self.states) + 1] + self.state_size, dtype=np.float32)

        accs = np.zeros([len(self.states) + 1, self.acc_size], dtype=np.float32)

        for i in range(len(self.states)):
            images[i], accs[i] = self.states[i]

        images[-1], accs[-1] = next_state

        states = [images, accs]

        values = self.critic2.predict(states)

        values = np.reshape(values, len(values))

        advantage = np.zeros_like(self.rewards, dtype=np.float32)

        gae = 0

        if done:
            values[-1] = np.float32([0])

        for t in reversed(range(len(self.rewards))):

            delta = self.rewards[t] + self.gamma * values[t + 1] - values[t]

            gae = delta + self.gamma * self.lambd * gae

            advantage[t] = gae

        target_val = advantage + values[:-1]

        target_val = target_val.reshape((-1, 1))

        advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-6)

        states = [images[:-1], accs[:-1]]

        actor_loss = self.actor_update(states + [self.actions, advantage])

        critic_loss = self.critic_update(states + [target_val])

        critic_loss = np.array(critic_loss[0])

        critic_loss = critic_loss[0]

        print(f'critic loss is {critic_loss}')

        self.clear_sample()

        return actor_loss[0], critic_loss

    def append_sample(self, state, action, reward):

        self.states.append(state)

        act = np.zeros(self.action_size)

        act[action] = 1

        self.actions.append(act)

        self.rewards.append(reward)

    def clear_sample(self):

        self.states.clear()

        self.actions.clear()

        self.rewards.clear()

    def update_target_model(self):

        self.critic2.set_weights(self.critic.get_weights())

    def load_model(self, name):

        if os.path.exists(name + '_best_actor.h5'):
            self.actor.load_weights(name + '_best_actor.h5')

            print('Actor loaded')

        if os.path.exists(name + '_best_critic.h5'):
            self.critic.load_weights(name + '_best_critic.h5')

            print('Critic loaded')

    def save_model(self, name):

        self.actor.save_weights(name + '_actor.h5')

        self.critic.save_weights(name + '_critic.h5')


'''

Environment interaction

'''


def transform_input(responses, img_height, img_width):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)

    img1d = np.array(np.clip(255 * 3 * img1d, 0, 255), dtype=np.uint8)

    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    image = Image.fromarray(img2d)

    image = np.array(image.resize((img_width, img_height)).convert('L'))

    image = np.float32(image.reshape(1, img_height, img_width, 1))

    image /= 255.0

    return image


def interpret_action(action):
    scaling_factor = 1.0

    forward_factor = 0.2

    if action == 0:

        quad_offset = (scaling_factor + forward_factor,0 ,-scaling_factor)

    elif action == 1:

        quad_offset = (scaling_factor + forward_factor, 0, 0)

    elif action == 2:

        quad_offset = (forward_factor, scaling_factor, 0)

    elif action == 3:

        quad_offset = (forward_factor, 0, scaling_factor)

    elif action == 4:

        quad_offset = (-scaling_factor + forward_factor, 0, 0)

    elif action == 5:

        quad_offset = (forward_factor, -scaling_factor, 0)

    elif action == 6:

        quad_offset = (forward_factor, 0, -scaling_factor)

    return quad_offset


if __name__ == '__main__':

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    agent_name = 'a2c'

    parser = argparse.ArgumentParser()

    parser.add_argument('--test', action='store_true')

    parser.add_argument('--img_height', type=int, default=72)

    parser.add_argument('--img_width', type=int, default=128)

    parser.add_argument('--actor_lr', type=float, default=5e-5)

    parser.add_argument('--critic_lr', type=float, default=5e-5)

    parser.add_argument('--gamma', type=float, default=0.999)

    parser.add_argument('--lambd', type=float, default=0.96)

    parser.add_argument('--entropy', type=float, default=5e-3)

    parser.add_argument('--updatetime', type=int, default=32)

    parser.add_argument('--seqsize', type=int, default=5)

    parser.add_argument('--target_rate', type=int, default=1000)

    args = parser.parse_args()

    if not os.path.exists('save_stat'):
        os.makedirs('save_stat')

    if not os.path.exists('save_model'):
        os.makedirs('save_model')

    # Make RL agent

    state_size = [args.seqsize, args.img_height, args.img_width, 1]

    action_size = 7

    agent = A2CAgent(

        state_size=state_size,

        action_size=action_size,

        actor_lr=args.actor_lr,

        critic_lr=args.critic_lr,

        gamma=args.gamma,

        lambd=args.lambd,

        entropy=args.entropy,

        updatetime=args.updatetime,

        load_model=True,

    )

    # Train

    episode = 0

    bias = np.linalg.norm(object_pos)


    stats = []

    env = windENV()

    if args.test:

        while True:

            try:

                done = False

                bug = False

                # stats

                timestep, score, pmax = 0, 0., 0.

                observe = env.reset()

                image, acc = observe

                try:

                    image = transform_input(image, args.img_height, args.img_width)

                except:
                    print('False Done')
                    continue

                history = np.stack([image] * args.seqsize, axis=1)

                acc = acc.reshape(1, -1)

                state = [history, acc]

                while not done:

                    timestep += 1

                    action, policy = agent.get_action(state)

                    real_action = interpret_action(action)

                    observe, reward, done, bias = env.step(real_action, bias)

                    image, acc = observe

                    try:

                        image = transform_input(image, args.img_height, args.img_width)

                    except:

                        bug = True

                        print(f'bug = {bug}')

                        break

                    history = np.append(history[:, 1:], [image], axis=1)

                    acc = acc.reshape(1, -1)

                    next_state = [history, acc]

                    # stats

                    pmax += float(np.amax(policy))

                    score += reward

                    state = next_state

                if bug:
                    continue

                pmax /= timestep

                # done

                print('Ep %d:  Step %d Score %.2f Pmax %.2f'

                      % (episode, timestep, score, pmax))

                episode += 1

            except KeyboardInterrupt:

                env.disconnect()

                break

    else:

        # Train

        time_limit = 500

        global_step = 0

        while True:

            try:

                done = False

                bug = False

                # stats

                bestS, timestep, score, pmax, acc_score = 0., 0, 0., 0., 0.

                t, actor_loss, critic_loss = 0, 0., 0.

                observe = env.reset()

                image, acc = observe

                image = transform_input(image, args.img_height, args.img_width)

                history = np.stack([image] * args.seqsize, axis=1)

                acc = acc.reshape(1, -1)

                state = [history, acc]

                while not done and timestep < time_limit:

                    t += 1

                    timestep += 1

                    global_step += 1

                    if global_step >= args.target_rate:
                        agent.update_target_model()

                        global_step = 0

                    action, policy = agent.get_action(state)

                    real_action = interpret_action(action)

                    print(f'real action is {real_action}')

                    observe, reward, done, bias = env.step(real_action, bias)

                    image, acc = observe

                    image = transform_input(image, args.img_height, args.img_width)

                    history = np.append(history[:, 1:], [image], axis=1)

                    acc = acc.reshape(1, -1)

                    acc_s = np.linalg.norm(acc)

                    next_state = [history, acc]

                    agent.append_sample(state, action, reward)

                    # stats

                    score += reward

                    acc_score += acc_s

                    pmax += float(np.amax(policy))

                    if t >= args.updatetime or done:

                        t = 0

                        a_loss, c_loss = agent.train_model(next_state, done)

                        actor_loss += float(a_loss)

                        critic_loss += float(c_loss)

                    state = next_state

                if bug:
                    continue

                # done

                pmax /= timestep

                actor_loss /= (timestep // args.updatetime + 1)

                critic_loss /= (timestep // args.updatetime + 1)

                acc_score /= (timestep // args.updatetime + 1)

                stats = [episode,  score, pmax, actor_loss, critic_loss, acc_score,timestep]

                # log stats

                if timestep > 5:

                    with open('save_stat/' + agent_name + '_stat.csv', 'a', encoding='utf-8', newline='') as f:

                        wr = csv.writer(f)

                        wr.writerow(['%.4f' % s if type(s) is float else s for s in stats])

                if np.max(score) < bestS:
                    highscore = bestS

                else:
                    bestS = np.max(score)
                    highscore = bestS
                    agent.save_model('./save_model/' + agent_name + '_best')

                agent.save_model('./save_model/' + agent_name)

                episode += 1

                print(episode)

                # tf.summary.scalar("score", score, step=episode)
                #
                # tf.summary.scalar("reward", reward, step=global_step)
                #
                # tf.summary.scalar("actor_loss", actor_loss, step=episode)
                #
                # tf.summary.scalar("critic_loss", critic_loss, step=episode)
                #
                # tf.summary.scalar("c_loss", c_loss, step=global_step)
                #
                # tf.summary.scalar("a_loss", a_loss, step=global_step)
                #
                # tf.summary.scalar("policy", pmax, step=episode)

            except KeyboardInterrupt:

                env.disconnect()

                break
