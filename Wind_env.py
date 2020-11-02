import time
import numpy as np
import airsim

#define destination
object_pos = [20,0,0]
#define boundary
outZ = [-50, 50]
outY = [-100,100]
Action_Space = ['00', '+x', '+y', '+z', '-x', '-y', '-z']

np.random.seed(10)

class windENV():

    def __init__(self):
        # Connect to the Airsim environment
        self.cl = airsim.MultirotorClient()
        self.cl.confirmConnection()
        self.action_size = 3
        self.duration = 0.25


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
        state_v = self.cl.getMultirotorState().kinematics_estimated.linear_velocity
        Img = self.cl.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        state_v = np.array([state_v.x_val, state_v.y_val, state_v.z_val])
        observation = [Img,state_v]
        return observation

    def step(self,speed):
        #Use givin action
        speed = [int(i) for i in speed]
        self.cl.simPause(False)
        self.cl.moveByMotorPWMsAsync(speed[0], speed[1], speed[2],speed[3], duration=self.duration)
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
        state = [(pos),(velocity),(angle_acc)]
        #define stop condition
        stop = pos.y_val < outY[0] or pos.y_val > outY[1] or pos.z_val < outZ[0] or pos.z_val > outZ[1]
        pos = np.array([pos.x_val,pos.y_val,pos.z_val],dtype=np.float)
        print(f'position is {pos}')
        bias = pos - object_pos
        success = np.linalg.norm(bias) < 10
        done = stop or success
        print(f'state = {done}')

        #compute reward
        reward = self.compute_reward(pos,velocity,angle_acc,stop)


        Img = self.cl.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        velocity = np.array([velocity.x_val,velocity.y_val,velocity.z_val])
        observation = [Img,velocity]
        return  observation, reward, done, state

    def compute_reward(self, pos, velocity, angle_acc,stop):
        velocity = np.array([velocity.x_val,velocity.y_val,velocity.z_val],dtype=np.float)
        speed = np.linalg.norm(velocity)
        angle_acc = np.array([angle_acc.x_val,angle_acc.y_val,angle_acc.z_val],dtype=np.float)
        weight_ar = 1.0
        weight_vr = 0.1
        if stop:
            reward = -10
        else:
            #define diiferent reward:
            speed_reward = 1*speed
            print(f'speed_reward = {speed_reward}')

            angle_acc_reward = 1./(np.linalg.norm(angle_acc)+1)
            print(f'angle_acc_reward = {angle_acc_reward}')
            reward = weight_ar*angle_acc_reward + weight_vr*speed_reward
        print(f'reward = {reward}')
        return reward

    def add_wind(self):
        w1 = np.random.randint(0,5)
        w2 = np.random.randint(0,5)
        w3 = np.random.randint(0,5)
        wind = airsim.Vector3r(w1,w2,w3)
        print(f'add wind vector = {wind}')
        self.cl.simSetWind(wind)

    def disconnect(self):
        self.cl.enableApiControl(False)
        self.cl.armDisarm(False)
        print('Disconnected')








