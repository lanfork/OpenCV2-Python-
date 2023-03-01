# read face
# Analyze how many times you yawn
# As soon as we detect the large separation between our upper and lower lips in our mouth landmarks,
# we count once for yawning

import dlib
import sys
import numpy as np
import cv2
import time

print("path for python", sys.executable)
print("dlib version", dlib.__version__)
# print("dlib path", dlib.__path__)
print("path of cv2", cv2.__path__)
print("version of cv2", cv2.__version__)

PREDICTOR_PATH = "images/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
print(type(detector))
print(type(predictor))

def get_landmarks(im):
    rects = detector(im, 1)

    # more than one face detected
    if len(rects) > 1:
        return "error"
    # no face detected
    if len(rects) == 0:
        return "error"

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0,0], point[0,1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0,0,255))
        cv2.circle(im, pos, 3, color=(0,255,255))
    return im

# get y_coordinate of  top_lip from our landmarks
def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    # shape of top_lip_pts (6, 1, 2)
    # top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    # shape of top_lip_mean (1, 2)
    # [[171.         236.66666667]]
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    # return the average of y coordinates of upper lip
    return int(top_lip_mean[:,1])

# get y_coordinate of lower lip
def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    # top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    # return the average of y coordinates of bottom lip
    return int(bottom_lip_mean[:, 1])

#
# starter code here
#

