# Face Recognition Live
# In this section, we use the VGGFace model to identify yourself (you'll need to include a picture of yourself
# in the directory stated in the code).
#
# It uses one-shot learning, so we only need one picture for our recognition system!

# The VGG-Face CNN descriptors are computed using our CNN implementation based on the VGG-Very-Deep-16 CNN architecture
# as described in [1] and are evaluated on the Labeled Faces in the Wild and the YouTube Faces dataset.

# place photos of people (one face visible) in the folder called ./person
# put our picture into person folder for testing on a webcam
# Faces are extracted using the haarcascade_frontface_default detector model
# Extracted faces are placed in the folder called "./group_of_faces"

# We are extracting the faces needed for our one-shot learning model, it will load 5 extracted faces

from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D
from keras.layers import MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import model_from_json


# Loading our HAARCascade Face Detector
face_detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

# Directory of image of persons we will extract faces from
mypath = "./person/"
image_file_names = [f for f in listdir(mypath) if isfile(join(mypath,f))]
print("collected image names")
print(image_file_names)

for image_name in image_file_names:
    person_image = cv2.imread(mypath + image_name)
    face_info = face_detector.detectMultiScale(person_image, 1.3, 5)
    for (x, y, w, h) in face_info:
        face = person_image[y:y+h, x:x+w]
        roi = cv2.resize(face, (128,128), interpolation= cv2.INTER_CUBIC)
    path = "./group_of_faces/" + "face_" + image_name
    cv2.imwrite(path, roi)
    cv2.imshow("face", roi)
    cv2.waitKey(2000)
cv2.destroyAllWindows()

# loads image from path and resizes it
# Notice that VGG model expects 224x224x3 sized input images.
# Here, 3rd dimension refers to number of channels or RGB colors.

# expanation of this code
# Keras works with batches of images. So, the first dimension is used for the number of
# samples (or images) you have.

# When you load a single image, you get the shape of one image, which is (size1,size2,channels).
# In order to create a batch of images, you need an additional dimension: (samples, size1,size2,channels)
# The preprocess_input function is meant to adequate your image to the format the model requires.
# Some models use images with values ranging from 0 to 1. Others from -1 to +1.
# Others use the "caffe" style, that is not normalized, but is centered.
def preprocess_image(image_path):
    # this PIL image instance
    img = load_img(image_path, target_size=(224,224))
    # convert to numpy array
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# loads the VGGFace model
def loadVggFaceModel():
    model = Sequential()
    # first block
    # adding zero on top, bottom, left and right
    model.add( ZeroPadding2D((1,1), input_shape=(224,224,3)) )
    # 64 3x3 filters with relu as the activation
    # for padding, I think the default padding is 1 (valid padding)
    # , and the default stride is 1.
    model.add( Convolution2D(64, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(64, (3,3), activation='relu') )
    model.add( MaxPooling2D((2,2), strides=(2,2)) )

    # second block
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(128, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(128, (3,3), activation='relu') )
    model.add( MaxPooling2D((2,2), strides=(2,2)) )

    # third block
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(256, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(256, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(256, (3, 3), activation='relu') )
    model.add( MaxPooling2D((2, 2), strides=(2, 2)))

    # fourth block
    model.add( ZeroPadding2D((1, 1)) )
    model.add( Convolution2D(512, (3, 3), activation='relu') )
    model.add( ZeroPadding2D((1, 1)) )
    model.add( Convolution2D(512, (3, 3), activation='relu') )
    model.add( ZeroPadding2D((1, 1)) )
    model.add( Convolution2D(512, (3, 3), activation='relu') )
    model.add( MaxPooling2D((2, 2), strides=(2, 2)) )

    # fifth block
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(512, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(512, (3,3), activation='relu') )
    model.add( ZeroPadding2D((1,1)) )
    model.add( Convolution2D(512, (3,3), activation='relu') )
    model.add( MaxPooling2D((2, 2), strides=(2, 2)) )

    model.add( Convolution2D(4096, (7,7), activation='relu') )
    model.add( Dropout(0.5) )
    model.add( Convolution2D(4096, (1,1), activation='relu') )
    model.add( Dropout(0.5) )
    # If you don't specify anything, no activation is applied
    # (ie. "linear" activation: a(x) = x).
    model.add( Convolution2D(2622, (1,1)) )
    model.add( Flatten() )
    model.add( Activation('softmax') )

    model.load_weights('vgg_face_weights.h5')
    print(model.summary())

    # we’ll use previous layer of the output layer for representation
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_descriptor


model = loadVggFaceModel()
print("Model Loaded")

# Vector Similarity
# We’ve represented input images as vectors. We will decide both pictures are
# same person or not based on comparing these vector representations.
# Now, we need to find the distance of these vectors.
# There are two common ways to find the distance of two vectors: cosine distance and euclidean distance.
# Cosine distance is equal to 1 minus cosine similarity. No matter which measurement we adapt,
# they all serve for finding similarities between vectors.
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# Test model using your webcom
# this codes looks up the faces you extracted in the "group_of_faces" and uses the similarity (Cosine
# similarity) to detect which faces is most similar to the one being extracted with your webcam

# points to your extracted faces
people_pictures = "./group_of_faces"

# this is dictionary with its key as person name, value as vector representation
all_people_faces = dict()

for file in listdir(people_pictures):
    person_face, extension = file.split(".")
    img = preprocess_image('./group_of_faces/%s.jpg' % (person_face))
    # we can represent images 2622 dimensional vector 
    face_vector = model.predict(img)[0,:]
    all_people_faces[person_face] = face_vector
    print(person_face, face_vector)

print("Face representations retrived successfully")

#
# Start coding here
#

