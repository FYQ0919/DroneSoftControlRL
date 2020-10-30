import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)

print("Setting wind to 10m/s in forward direction") # NED
wind = airsim.Vector3r(5, 0, 0)
client.simSetWind(wind)
client.moveToPositionAsync(20,0,20,5).join()
# Takeoff or hover
landed = client.getMultirotorState().landed_state
if landed == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    print("already flying...")
    client.hoverAsync().join()

time.sleep(5)

print("Setting wind to 15m/s towards right") # NED
wind = airsim.Vector3r(3, 5, 0)
client.simSetWind(wind)

time.sleep(50)

# Set wind to 0
print("Resetting wind to 0")
wind = airsim.Vector3r(30, 30, 3)
client.simSetWind(wind)
