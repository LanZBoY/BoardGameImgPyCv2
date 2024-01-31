import cv2
import os 
import numpy as np
from cvutills import *
# BLOCK_SIZE = 100

ROAD2PAT = [None for _ in range(11)]
RESOURCE = '.\\resource'
TEST_MODE = 1 # 0 for frame, 1 for pictue
VERSION = "penguin" # penguin | space
PATTERN_PATH = os.path.join(RESOURCE, VERSION)
SAVE_PATH = '.\\procedure_img'
TRACE_MODE = True
ROI_SIDE_LENGTH = 600

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

trace_imgs = {}

if TEST_MODE == 0:
    bgr, trace_imgs = capture_roi_frame(TRACE_MODE, SAVE_PATH, ROI_SIDE_LENGTH, (1280, 720), version=VERSION)
    cv2.imwrite('wrap.jpg', trace_imgs['bgr'])
elif TEST_MODE == 1:
        bgr = cv2.imread('wrap.jpg')

bgr = bgr[9:-9, 15:-5] # range to detect
cv2.imshow("SRC", bgr)
# hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL)
# ### init detect tool

perfect_pattern = ROAD2PAT[road_type['BLACK']].copy()
if VERSION == 'space':
    perfect_pattern = cv2.inRange(perfect_pattern, np.array([85, 80, 80]), np.array([255, 255, 255]))
else:
    perfect_pattern = cv2.inRange(perfect_pattern, np.array([0, 0, 0]), np.array([120, 120, 80]))

if (VERSION == 'space'):
    detect_result = detect_space(bgr = bgr, perfect_pattern = perfect_pattern, road_type = road_type, TRACE_MODE = TRACE_MODE, trace_imgs = trace_imgs)
else:
    detect_result = detect_penguin(bgr = bgr, perfect_pattern = perfect_pattern, road_type = road_type, TRACE_MODE = TRACE_MODE, trace_imgs = trace_imgs)


project_on_board(detect_result, board)
cv2.imshow("final_board", board)

if TRACE_MODE:
    trace_imgs['final_board'] = board
    trace_imgs : dict
    for key, array in trace_imgs.items():
        print(f"Writing to ./procedure_img/{key}.jpg")
        cv2.imwrite(f'./procedure_img/{key}.jpg', array)


cv2.waitKey(0)
cv2.destroyAllWindows()