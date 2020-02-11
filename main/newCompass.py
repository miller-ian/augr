import sys, getopt

sys.path.append('.')
import RTIMU
import os.path
import time
import math


SETTINGS_FILE = "RTIMULib"

print("Using settings file " + SETTINGS_FILE + ".ini")
if not os.path.exists(SETTINGS_FILE + ".ini"):
  print("Settings file does not exist, will be created")

s = RTIMU.Settings(SETTINGS_FILE)
imu = RTIMU.RTIMU(s)


print("IMU Name: " + imu.IMUName())

if (not imu.IMUInit()):
    print("IMU Init Failed")
    sys.exit(1)
else:
    print("IMU Init Succeeded")

imu.setCompassEnable(True)

while True:
    if imu.IMURead():
        data = imu.getIMUData()
        compassData = data["compass"]
        compassX = compassData[0]
        compassY = compassData[1]
        gaussX = 1
        gaussY = 1
        if compassX != 0:
            gaussX = compassX
        if compassY != 0:
            gaussY = compassY
        bearing = math.atan2(gaussY, gaussX)
        bearing = math.degrees(bearing)
     
        print(bearing)
