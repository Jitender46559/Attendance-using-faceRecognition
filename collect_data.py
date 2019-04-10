import cv2
import numpy as np
import random

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

i=0
while(i!=5000):

    ret, frame = cap.read()
    i+=1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.5, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame_gray[y:y + h, x:x + w]     # region of interest
        roi_color = frame[y:y + h, x:x + w]
    cv2.imwrite('C:/Users/Jitender kumar/PycharmProjects/Face_recognition/Images/Jitender Kumar/{index}.jpg'.format(index=i), roi_gray)

    cv2.imshow('Collecting data', roi_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()