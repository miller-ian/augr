from imutils import paths
import numpy as np
import imutils
import cv2

focalLength = 4 # mm
realHeight = 300 # mm
imagePixels = 600 # width
objectPixels = 0
sensorHeight = 12 # mm

def calculate_distance(focalLength, realHeight, imagePixels, objectPixels, sensorHeight):
    distance = (focalLength * realHeight * imagePixels) / (objectPixels * sensorHeight)
    return distance / 25.4

eyeCascade = cv2.CascadeClassifier('eye_cascade.xml')
faceCascade = cv2.CascadeClassifier('face_cascade.xml')

def detect(gray, frame):
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 3)

        # h is the number of pixels the y dimension of the face-box is
        print("distance from camera to head: ", calculate_distance(focalLength, realHeight, imagePixels, h, sensorHeight))
        
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return frame

videoCap = cv2.VideoCapture(0)
while True:
    _, frame = videoCap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
videoCap.release()
cv2.destroyAllWindows()
