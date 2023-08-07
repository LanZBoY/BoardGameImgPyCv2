
import numpy as np
import cv2

min_0_thr = 0
min_1_thr = 0
min_2_thr = 0
max_0_thr = 255
max_1_thr = 255
max_2_thr = 255

src = cv2.imread("result.jpg")
# src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV_FULL)

def change_min_0(val):
    global min_0_thr
    min_0_thr = val
    onChange()

def change_min_1(val):
    global min_1_thr
    min_1_thr = val
    onChange()

def change_min_2(val):
    global min_2_thr
    min_2_thr = val
    onChange()

def change_max_0(val):
    global max_0_thr
    max_0_thr = val
    onChange()

def change_max_1(val):
    global max_1_thr
    max_1_thr = val
    onChange()

def change_max_2(val):
    global max_2_thr
    max_2_thr = val
    onChange()

def onChange():
    global src, min_0_thr, min_1_thr, min_2_thr, max_0_thr, max_1_thr, max_2_thr
    result = cv2.inRange(src, np.array([min_0_thr, min_1_thr, min_2_thr]), np.array([max_0_thr, max_1_thr, max_2_thr]))
    cv2.imshow("result", result)

cv2.namedWindow("ControlPannel")
"""
BLANK_DETECT BGR
 B = [0, 140]
 G = [0, 120]
 R = [0 , 60]

RED_DETECT HSV
 R_H = [0 - 50]
 
GREEN_DETECT HSV
 G_H = [40 - 100]
"""
cv2.imshow('src', src)
cv2.createTrackbar('min_0_thr','ControlPannel', 0, 255, change_min_0)
cv2.createTrackbar('max_0_thr','ControlPannel', 0, 255, change_max_0)
cv2.createTrackbar('min_1_thr','ControlPannel', 0, 255, change_min_1)
cv2.createTrackbar('max_1_thr','ControlPannel', 0, 255, change_max_1)
cv2.createTrackbar('min_2_thr','ControlPannel', 0, 255, change_min_2)
cv2.createTrackbar('max_2_thr','ControlPannel', 0, 255, change_max_2)
cv2.waitKey(0)
cv2.destroyAllWindows()