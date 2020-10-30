import time
import numpy as np
import airsim
import config

object_pos = [200,0,10]

class windENV():

    def __init__(self):
        # Connect to the Airsim environment
        self.cl = airsim.MultirotorClient()
        self.cl.confirmConnection()
        self.action_size = 5
        self.clock = 1.5
        self.duration = 0.85/self.clock


    def reset(self):
        self.cl.reset()
        self.cl.enableApiControl(True)
        self.cl.armDisarm(True)

        #take off
        self.cl.simPause(False)
        self.cl.takeoffAsync().join()
        self.cl.hoverAsync().join()
        self.cl.simPause(True)
        state_v = self.cl.getMultirotorState().kinematics_estimated.linear_velocity
        Angle_acc = self.cl.getMultirotorState().kinematics_estimated.angular_acceleration
        state_v = np.array([state_v.x_val, state_v.y_val, state_v.z_val])
        observation = [Angle_acc,state_v]
        return observation

    def step(self,speed):
        #Use givin action
        speed = [int(i) for i in speed]
        self.cl.simPause(False)
        self.cl.moveByMotorPWMsAsync(speed[0], speed[1], speed[2],speed[3],speed[4],duration=self.duration)
        start = time.time()
        while time.time() - start < self.duration:
            #get states
            pos = self.cl.getMultirotorState().kinematics_estimated.position
            velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.cl.simPause(True)
        Img = self.cl.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])

        # get states
        pos = self.client.getMultirotorState().kinematics_estimated.position
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        dead =







