from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

'''
When optical flow detector returns movement, this file will be run.
When this file is running, it will be making detections in real-time.
A final script will be run sending messages to server at a predefined 
frequency to populate ATAK. 
'''

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model...")

model = "mobilenet.caffemodel"
prototxt = "mobilenet.txt"
confidence = 0.3
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:

    #grab frame and resize to width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)

    #grab frame dimensions and convert frame to blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    #pass blob through network and obtain the detections
    net.setInput(blob)
    detections = net.forward() 
    for i in np.arange(0, detections.shape[2]):

		# extract the confidence (i.e., probability) associated with
		# the prediction
        detectionConfidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if detectionConfidence > confidence:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], detectionConfidence * 100)
            print(label)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    #cv2.imshow("Frame", frame)
    #key = cv2.waitKey(1) & 0xFF
    # update the FPS counter
    fps.update()

fps.stop()


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()