"""
Input is a video or video source such as a webcam. 
This program extracts images from video at a specified rate per second (fps) and saves them to the root directory.
"""

import numpy as np
import cv2
import time
import droidDetector

cap = cv2.VideoCapture(0)
count = 0

while True:
    _, frame = cap.read()
    theCount = str(count)
    filename = theCount + ".jpg"
    cv2.imwrite(filename, frame)
    droidDetector.detect(theCount)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break