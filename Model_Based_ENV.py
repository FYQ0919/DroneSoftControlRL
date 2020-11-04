import time
import numpy as np
import airsim

#define destination
object_pos = [13,0,1]
#define boundary
outZ = [-5, 2]
outY = [-5,5]
outX = [-5,15]

np.random.seed(10)

class windENV():

    def __init__(self):
        # Connect to the Airsim environment
        self.cl = airsim.MultirotorClient()
        self.cl.confirmConnection()
        self.action_size = 3
        self.duration = 0.2


    def reset(self):
        self.cl.reset()
        self.cl.enableApiControl(True)
        self.cl.armDisarm(True)
        self.add_wind()
        time.sleep(5)

        #take off
        self.cl.simPause(False)
        self.cl.takeoffAsync().join()
        self.cl.hoverAsync().join()
        self.cl.simPause(True)
        state_acc = self.cl.getMultirotorState().kinematics_estimated.angular_acceleration
        vel =  self.cl.getMultirotorState().kinematics_estimated.linear_velocity
        Img = self.cl.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        state_acc = np.array([state_acc.x_val, state_acc.y_val, state_acc.z_val])
        vel = np.array([vel.x_val, vel.y_val, vel.z_val])
        observation = [Img,state_acc]
        return observation

    def step(self,speed,bias):
        #Use givin action
        speed = [int(i) for i in speed]
        self.cl.simPause(False)
        self.cl.moveByVelocityAsync(speed[0], speed[1], speed[2] , duration=self.duration)
        start = time.time()
        while time.time() - start < self.duration:
            #Add Wind Noise
            self.add_wind()
            time.sleep(self.duration)

            #get states
            pos = self.cl.getMultirotorState().kinematics_estimated.position
            velocity = self.cl.getMultirotorState().kinematics_estimated.linear_velocity
            angle_acc = self.cl.getMultirotorState().kinematics_estimated.angular_acceleration

        self.cl.simPause(True)


        # update states
        pos = self.cl.getMultirotorState().kinematics_estimated.position
        velocity = self.cl.getMultirotorState().kinematics_estimated.linear_velocity
        angle_acc = self.cl.getMultirotorState().kinematics_estimated.angular_acceleration

        #define stop condition
        stop = pos.y_val < outY[0] or pos.y_val > outY[1] or pos.z_val < outZ[0] or pos.z_val > outZ[1] or pos.x_val < outX[0] or pos.x_val > outX[1]
        pos = np.array([pos.x_val,pos.y_val,pos.z_val],dtype=np.float)
        print(f'position is {pos}')
        new_bias = pos - object_pos
        success = np.linalg.norm(bias) < 1
        done = stop or success
        if stop:
            print('state = stop')
        elif success:
            print('state = success')


        #compute reward
        reward = self.compute_reward(velocity,angle_acc,stop,bias,object_pos,new_bias,success)



        Img = self.cl.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        angle_acc = np.array([angle_acc.x_val,angle_acc.y_val,angle_acc.z_val])
        observation = [Img,angle_acc]
        bias = new_bias
        return  observation, reward, done ,bias

    def compute_reward(self, velocity, angle_acc,stop,bias,object_pos,new_bias,success):

        velocity = np.array([velocity.x_val,velocity.y_val,velocity.z_val],dtype=np.float)
        speed = np.linalg.norm(velocity)
        angle_acc = np.array([angle_acc.x_val,angle_acc.y_val,angle_acc.z_val],dtype=np.float)
        standard_dis = np.linalg.norm(object_pos)
        distance =  np.linalg.norm(bias)
        new_distance =  np.linalg.norm(new_bias)
        weight_ar = 1.0
        weight_dis = 0.5
        weight_vr = 0.1

        if stop:
            reward = -50
        elif success:
            reward = 500
        else:
            #define diiferent reward:

            speed_reward = 1*speed
            print(f'speed_reward = {speed_reward}')

            if distance < standard_dis and new_distance < distance:
                distance_reward = 1.
            else:
                distance_reward = -1.5
            print(f'distance reward is {distance_reward}')

            angle_acc_reward = 1./(np.linalg.norm(angle_acc)+1)
            print(f'angle_acc_reward = {angle_acc_reward}')

            reward = weight_ar*angle_acc_reward + weight_vr*speed_reward + weight_dis*distance_reward
        print(f'reward = {reward}')
        return reward

    def add_wind(self):
        w1 = np.random.randint(0,3)
        w2 = np.random.randint(0,3)
        w3 = np.random.randint(0,3)
        wind = airsim.Vector3r(w1,w2,w3)
        print(f'add wind vector = {wind}')
        self.cl.simSetWind(wind)

    def disconnect(self):
        self.cl.enableApiControl(False)
        self.cl.armDisarm(False)
        print('Disconnected')








