import time
import serial
import string
import pynmea2
import RPi.GPIO as gpio

gpio.setmode(gpio.BCM)

port = "/dev/ttyAMA0"
ser = serial.Serial(port, baudrate = 9600, timeout = 1)
msg = ''
latVal = ''
lonVal = ''

while True:
	try:
		data = ser.readline()
	except:
		
		print("loading")
	
	if data[0:6] == '$GPGLL':
		msg = pynmea2.parse(data)
		latVal = msg.lat
		lonVal = msg.lon
	print("latitude: ", str(latVal))
	print("longitude: ", str(lonVal))

