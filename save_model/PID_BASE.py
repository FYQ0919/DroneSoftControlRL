import time
import numpy as np
import airsim
object_pos = [15,0,1]


def add_wind():
    w1 = np.random.randint(3, 5)
    w2 = np.random.randint(3, 5)
    w3 = np.random.randint(3, 5)
    wind = airsim.Vector3r(w1, w2, w3)
    print(f'add wind vector = {wind}')
    client.simSetWind(wind)
    time.sleep(5)

client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)

Angle_acc = 0
client.simPause(False)
client.takeoffAsync().join()
client.hoverAsync().join()
client.simPause(True)
for i in  range(10):
    client.simPause(False)
    add_wind()
    client.moveToPositionAsync(15,0,1,0.2)
    angle_acc = client.getMultirotorState().kinematics_estimated.angular_acceleration
    Angle_acc += 32*np.linalg.norm(np.array([angle_acc.x_val,angle_acc.y_val,angle_acc.z_val],dtype=np.float))
    client.simPause(True)


print(Angle_acc)



