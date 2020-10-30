import time
import numpy as np
import airsim

#define destination
object_pos = [200,0,10]
#define boundary
outZ = [-5, 20]
outY = [-10,10]
Action_Space = ['00', '+x', '+y', '+z', '-x', '-y', '-z']

np.random.seed(10)

class windENV():

    def __init__(self):
        # Connect to the Airsim environment
        self.cl = airsim.MultirotorClient()
        self.cl.confirmConnection()
        self.action_size = 5
        self.clock = 5
        self.duration = 0.5/self.clock


    def reset(self):
        self.cl.reset()
        self.cl.enableApiControl(True)
        self.cl.armDisarm(True)
        self.add_wind()

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
        self.cl.moveByVelocityAsync(speed[0], speed[1], speed[2],duration=self.duration)
        start = time.time()
        while time.time() - start < self.duration:
            #get states
            pos = self.cl.getMultirotorState().kinematics_estimated.position
            velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            angle_acc = self.cl.getMultirotorState().kinematics_estimated.angular_acceleration
        self.cl.simPause(True)


        # update states
        pos = self.client.getMultirotorState().kinematics_estimated.position
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        angle_acc = self.cl.getMultirotorState().kinematics_estimated.angular_acceleration
        state = [(pos),(velocity),(angle_acc)]
        #define stop condition
        stop = pos.y_val < outY[0] or pos.y_val > outY[1] or pos.z_val < outZ[0] or pos.z_val > outZ[1]
        bias = pos - object_pos
        success = np.linalg.norm(bias) < 10
        done = stop or success

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
        weight_mr = 0.5
        weight_ar = 1.0
        weight_vr = 0.5
        if stop:
            reward = -10
        else:
            #define diiferent reward:
            move_reward =  100./np.linalg.norm(pos-object_pos)
            speed_reward = speed
            angle_acc_reward = 100./np.linalg.norm(speed)
            reward = weight_ar*angle_acc_reward + weight_mr*move_reward + weight_vr*speed_reward

        return reward

    def add_wind(self):
        w1 = np.random.uniform(0,10,int)
        w2 = np.random.uniform(0,10,int)
        w3 = np.random.uniform(0,10,int)
        wind = airsim.Vector3r(w1,w2,w3)
        self.cl.simSetWind(wind)

    def disconnect(self):
        self.cl.enableApiControl(False)
        self.cl.armDisarm(False)
        print('Disconnected')








