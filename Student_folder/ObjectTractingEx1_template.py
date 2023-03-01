# Lucas-Kanade Optical Flow in OpenCV
# check the slides for details of paramter
# OpenCV provides all these in a single function, cv.calcOpticalFlowPyrLK().
# Here, we create a simple application which tracks some points in a video.


import numpy as np
import cv2

# load video stream
cap = cv2.VideoCapture('images/walking.avi')
# cap = cv2.VideoCapture('images/cars.avi')



# OpenCV has a function, cv2.goodFeaturesToTrack().
# It finds N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection,
# if you specify it).

# Set parameters for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)

# set parameters for lucas kanade optical flow
lucas_kanade_params = dict( winSize = (15,15),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#
# start coding here
#

