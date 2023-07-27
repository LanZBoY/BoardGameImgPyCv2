from cvutills import capture_roi
import numpy as np
import cv2
from datamodel import ROAD_TYPE_LIST

min_thr = 0
max_thr = 255
channel = 0
def get_block(warp : np.ndarray, x : int, y : int, BLOCK_SIZE = 100):
    return warp[y * BLOCK_SIZE:(y + 1) * BLOCK_SIZE - 1, x * BLOCK_SIZE:(x + 1) * BLOCK_SIZE - 1].copy()

src = cv2.imread("result.jpg")
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

def change_min(val):
    global min_thr
    min_thr = val
    onChange()

def change_max(val):
    global max_thr
    max_thr = val
    onChange()

def change_channel(val):
    global channel
    channel = val
    onChange()

def onChange():
    global src, channel
    temp = src[:,:]
    result = cv2.inRange(temp, min_thr, max_thr)
    cv2.imshow("result", result)

# print("Initializing data...")
# ROAD_TO_MAT = {
#     "UNKNOWN" : None, 
#     "UNPROCESS" : None, 
#     "START" : None, 
#     "BLACK" : None, 
#     "BLUE" : None, 
#     "RED" : None, 
#     "GREEN" : None, 
#     "BLUE" : None, 
#     "VERTICAL" : None, 
#     "HORIZON" : None, 
#     "UP_LEFT" : None, 
#     "UP_RIGHT" : None, 
#     "LEFT_DOWN" : None, 
#     "RIGHT_DOWN" : None}

# for ROAD in ROAD_TYPE_LIST[2:]:
#     data = cv2.imread(f'.\\resource\\{ROAD}.png')
#     data = cv2.resize(data, (100, 100), cv2.INTER_NEAREST)
#     ROAD_TO_MAT[ROAD] = data

cv2.namedWindow("result")
"""
 R -> [0-50]
 G -> [40 - 85]
 B -> 
"""
cv2.createTrackbar('min_thr','result', 0, 255, change_min)
cv2.createTrackbar('max_thr','result', 0, 255, change_max)
cv2.createTrackbar('chanel','result', 0, 2, change_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()