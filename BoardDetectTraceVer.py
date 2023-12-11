import cv2
import os 
import numpy as np
from cvutills import *
# BLOCK_SIZE = 100



def project_on_board(detect_result, board):
    for i in range(7):
        for j in range(7):
            if detect_result[j, i] == -1:
                continue
            set_block(board[18:-18, 18:-18], i, j, ROAD2PAT[detect_result[j, i]])

road_type = {
        "BLANK" : 0,
        "RED" : 1,
        "GREEN" : 2,
        "BLUE" : 3,
        "BLACK" : 4,
        "HORIZON" : 5,
        "VERTICAL" : 6,
        "UP_RIGHT" : 7,
        "DOWN_LEFT" : 8,
        "UP_LEFT" : 9,
        "RIGHT_DOWN" : 10,
        "BOARD" : None,
        # "AGENT" : 11,
        "UNPROCESS": -1
    }

ROAD2PAT = [None for _ in range(11)]
RESOURCE = '.\\resource'
TEST_MODE = 0 # 0 for frame, 1 for pictue
VERSION = "penguin" # penguin | space
PATTERN_PATH = os.path.join(RESOURCE, VERSION)


for key in road_type.keys():
    file_path = os.path.join(PATTERN_PATH, f'{key}.png')
    if os.path.isfile(file_path):
        res = cv2.imread(file_path)
        if key == "BOARD":
            board = cv2.resize(res, (736,736))
        else :
            pattern = cv2.resize(res, (100,100))
            ROAD2PAT[road_type[key]] = pattern
# ROAD2PAT[road_type['BLANK']] = np.zeros_like(pattern)
SAVE_PATH = '.\\procedure_img'
TRACE_MODE = True
ROI_SIDE_LENGTH = 600

if TEST_MODE == 0:
    bgr, trace_imgs = capture_roi_frame(TRACE_MODE, SAVE_PATH, ROI_SIDE_LENGTH, (1280, 720), version=VERSION)
    cv2.destroyAllWindows()
elif TEST_MODE == 1:
    if VERSION == 'space':
        bgr = cv2.imread('space_result.jpg')
    else:
        bgr = cv2.imread('penguin_result.jpg')
cv2.imshow("SRC", bgr)
cv2.imwrite('crop.jpg', trace_imgs['crop'])

bgr = bgr[18:-18, 18:-18] # range to detect
# hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL)
# ### init detect tool

perfect_pattern = ROAD2PAT[road_type['BLACK']].copy()
if VERSION == 'space':
    perfect_pattern = cv2.inRange(perfect_pattern, np.array([85, 80, 80]), np.array([255, 255, 255]))
else:
    perfect_pattern = cv2.inRange(perfect_pattern, np.array([0, 0, 0]), np.array([120, 120, 80]))

if (VERSION == 'space'):
    detect_result = detect_space(bgr = bgr, perfect_pattern = perfect_pattern, road_type = road_type)
else:
    detect_result = detect_penguin(bgr = bgr, perfect_pattern = perfect_pattern, road_type = road_type)
    
cv2.waitKey(0)
cv2.destroyAllWindows()

project_on_board(detect_result, board)
cv2.imshow("Result", board)
cv2.waitKey(0)
cv2.destroyAllWindows()