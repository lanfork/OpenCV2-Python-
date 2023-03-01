# Run our facial recognition
# 3/10/2020

# Successfully installed dlib-19.16.0
# Train Model
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import time
# /home/jetzhong/anaconda3/bin/python
print(sys.executable)

print(cv2.__path__)
print(cv2.__version__)

# Get the training data we previously made
data_path = './faces/user/'

# onlyfiles will store the list of file name in the directory
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

# extract faces from webcam view
def face_detector(img, size=0.5):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # print(faces)
    if faces is ():
        return img, []
    # assume we only have single face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img, roi

# test our facial recognition classfier
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

#
# starter code here
#
