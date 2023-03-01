from keras.models import load_model
import cv2
import numpy as np
from time import sleep
from keras_preprocessing.image import img_to_array
# Loading our saved model
classifier = load_model('./TrainedModels/emotion_little_vgg_3.h5')

# Haarcascades is the type of object detector ( we will use face detection capability)
face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')

# this function will return rectangles of location of face, cropped image of face and
# complete image
def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return array of locations of faces
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # if no faces, return some default value
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    # (x, y, w, h) is the location of the face
    for (x, y, w, h) in faces:
        # draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # crop the image to face out of the image
        # we need tightly cropped face
        roi_gray = gray[y:y + h, x:x + w]

    try:
        # resize roi_gray to the image which classifier accepts
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    except:
        return (x, w, y, h), np.zeros((48, 48), np.uint8), img
    return (x, w, y, h), roi_gray, img

# Get our class labels
import keras
from keras.preprocessing.image import ImageDataGenerator

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_data_dir = './fer2013/validation'
num_classes = 6
img_rows, img_cols = 48, 48
batch_size = 16

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode = 'grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print(class_labels)

#
# starter code here
#
