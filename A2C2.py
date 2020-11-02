#Author: Fu Yangqing
#NUS ID: A0225413R
#Description:
#Use A2C algorithm to train quadrotor to do soft motion
#Update time:2020/10/30


import time
import random as rd
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, ZeroPadding2D, Dense, Dropout, Activation, Convolution2D, Reshape
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
import csv
import cv2
agent_name = 'A2C'






class A2CAgent(object):

    def __init__(self, state_size, action_size, actor_lr, critic_lr,

                 gamma, lambd, entropy, horizon, load_model):

        self.state_size = state_size

        self.action_size = action_size

        self.vel_size = 3

        self.acc_size = 3

        self.actor_lr = actor_lr

        self.critic_lr = critic_lr

        self.gamma = gamma

        self.lambd = lambd

        self.entropy = entropy

        self.horizon = horizon

        self.actor, self.critic = self.build_model()

        _, self.target_critic = self.build_model()

        self.actor_update = self.build_actor_optimizer()

        self.critic_update = self.build_critic_optimizer()

        if load_model:
            self.load_model('./save_model/' + agent_name)

        self.target_critic.set_weights(self.critic.get_weights())

        self.states, self.actions, self.rewards = [], [], []

class A2CModel(tf.keras.Model):
    def __init__(self, img_size):
        super(A2CModel, self).__init__()

        self.tfd = tfp.distributions
        self.img_size = img_size
        self._build_net()

    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs, name='img_input')

        # trunk------------------
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.mp1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.mp2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.mp3(x)
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.mp4(x)
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.mp5(x)
        # trunk-----------------
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        m = self.dense_mean(x)
        # v1 = layers.Dense(1, activation='sigmoid', name='variance1')(x)
        # v2 = layers.Dense(1, activation='sigmoid', name='variance2')(x)
        # distribution_param = tf.squeeze(tf.concat([m1, m2, v1, v2], axis=0))
        m = tf.squeeze(m)
        tf.print('means:', m)
        action_onehot = self.distribution(m)

        value = self.dense_value(x)
        # todo: try covariance matrices
        # nd = self.tfd.MultivariateNormalDiag(loc=[m1, m2], scale_diag=[v1, v2])
        # possibility_map = tf.squeeze(nd.prob(self.sample_list))
        # action_map = tf.reshape(tf.nn.softmax(possibility_map), [self.img_size, self.img_size])
        return m, action_onehot, value

    def _build_net(self):
        # build net----------------------------------
        self.conv11 = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv12 = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.mp1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv21 = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv22 = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.mp2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv31 = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv32 = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.mp3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv41 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv42 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.mp4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv51 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv52 = layers.Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')
        self.mp5 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten = layers.Flatten()

        self.dense1 = layers.Dense(2048, activation='relu')
        self.dense2 = layers.Dense(2048, activation='relu')
        self.dense_mean = layers.Dense(2, activation='sigmoid', name='mean')

        self.dense_value = layers.Dense(1, activation='linear', name='value')
        self.distribution = ActionMap(self.img_size)
