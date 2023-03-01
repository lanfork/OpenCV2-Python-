#  Deep face version


# this is deepFaceReader combine emotion, age and gender recognition
# at first we will detect face, then find out emotion (facial expression),
# estimated age of person and predicted gender

############
# homework
# testing image with multiple face and try dlib ???
# dlib is better than Hascade for facereader
######################################


from pathlib import Path
import cv2
import dlib
import sys
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
import tensorflow
import keras
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import time

print("path for python")
print(sys.executable)
print()

print("cv2")
print(cv2.__path__)
print(cv2.__version__)
print()

print("dlib")
print(dlib.__version__)
print()

print("numpy")
print(np.__path__)
print(np.__version__)
print()

print('tensorflow')
print(tensorflow.__path__)
print('tensorflow: ' + tensorflow.__version__)
print()

print("keras")
print(keras.__path__)
print(keras.__version__)
print()

# this is the classifier for emotion detection using little VGG model
# this model 60% accuracy
classifier = load_model('emotion_little_vgg_2.h5')

# load our Harrcascade classifier for detecting face
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Load our pretrained model for Gender and Age detection
pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
print(pretrained_model)
print()

# We need specify modhash when we load the model
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

# We have 6 facial expression
emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
print("type of emotion")
print(emotion_classes)
print()
print()

# Face Detection function
def face_detector(img):
    # convert image to grayscale for faster detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # array of face locations
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return False, (0,0,0,0), np.zeros((1,48,48,3), np.uint8), img

    # images of only faces (roi)
    allfaces = []
    # locations of faces
    rects = []
    # x, y, w, h is location of face
    for (x,y,w,h) in faces:
        # put rectangle around the image
        cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),2)
        # crop face regsion
        roi = img[y:y+h, x:x+w]
        allfaces.append(roi)
        rects.append((x,w,y,h))
    return True, rects, allfaces, img

# define our model parameters for age and gender detector
depth = 16
k = 8
weight_file = None
margin = 0.4
image_dir = None

# Get our weight_file
if not weight_file:
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                           file_hash=modhash, cache_dir=Path(sys.argv[0]).resolve().parent)

# load model and weights for age and gender detection which is WideResNet
img_size = 64
model = WideResNet(img_size, depth=depth, k=k) ()
model.load_weights(weight_file)
print(model)

#
# start coding here
#

