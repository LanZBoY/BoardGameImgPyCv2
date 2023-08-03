import cv2
import os 
import numpy as np
from cvutills import *
# BLOCK_SIZE = 100
def get_block(mat, x, y):
    return mat[y * 100 : (y * 100) + 100, x * 100 : (x * 100) + 100]

def set_block(mat, x, y, pattern):
    try:
        mat[y * 100 : (y * 100) + 100, x * 100 : (x * 100) + 100] = pattern
    except:
        pass

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
        "UNPROCESS": -1
    }


def get_road(array):
    global road_type, ROAD2PAT
    if array[0] and array[1] and array[2] and array[3]:
        return road_type['BLACK']
    elif array[0] and array[1] and ~array[2] and ~array[3]:
        return road_type['UP_RIGHT']
    elif ~array[0] and array[1] and array[2] and ~array[3]:
        return road_type['RIGHT_DOWN']
    elif ~array[0] and ~array[1] and array[2] and array[3]:
        return road_type['DOWN_LEFT']
    elif array[0] and array[1] and ~array[2] and ~array[3]:
        return road_type['UP_LEFT']
    elif array[0] and ~array[1] and array[2] and ~array[3]:
        return road_type['VERTICAL']
    elif ~array[0] and array[1] and ~array[2] and array[3]:
        return road_type['HORIZON']
    return road_type['UNPROCESS']

ROAD2PAT = [None for _ in range(11)]
RESOURCE = '.\\resource'
VERSION = "space" # penguin | space
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

SAVE_PATH = '.\\procedure_img'
TRACE_MODE = False
ROI_SIDE_LENGTH = 600

roi_img, trace_imgs = capture_roi_frame(TRACE_MODE, SAVE_PATH, ROI_SIDE_LENGTH, (1280, 720), version=VERSION)
# cv2.destroyAllWindows()
# gray_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
# gray_img = cv2.GaussianBlur(gray_img, ksize=(7, 7), sigmaX=0)
# _, otsu = cv2.threshold(gray_img, 0., 255., cv2.THRESH_OTSU)
# cv2.imshow('roi', roi_img)
# cv2.imshow("otsu", otsu)

cv2.imshow("space_result", roi_img)
cv2.imwrite('space_result.jpg', roi_img)
# roi_img = roi_img[18:-18, 18:-18]
# hsv_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV_FULL)

# ### init detect tool
# perfect_pattern = ROAD2PAT[road_type['BLACK']].copy()
# perfect_pattern = cv2.inRange(perfect_pattern[:, :, 0], 0, 80) | cv2.inRange(perfect_pattern[:, :, 1], 0, 80) | cv2.inRange(perfect_pattern[:, :, 2], 0, 80)
# detect_result = np.zeros(shape=(7, 7), dtype=np.int8)
# detect_result.fill(-1)

# road_detect = cv2.inRange(roi_img, np.array([0, 0, 0]), np.array([115, 255, 255]))
# cv2.imshow('road', road_detect)
# blank_detect = ~(road_detect)
# for i in range(7):
#     for j in range(7):
#         if detect_result[j, i] != -1:
#             continue
#         detect = get_block(blank_detect, i, j)
#         score = np.count_nonzero(detect) / (detect.shape[0] * detect.shape[1])
#         if score >= .95:
#             detect_result[j, i] = road_type['BLANK']

# red_detect = cv2.inRange(hsv_img[:,:,0], 0, 50)
# cv2.imshow('red', red_detect)
# for i in range(7):
#     for j in range(7):
#         if detect_result[j, i] != -1:
#             continue
#         detect = get_block(red_detect, i, j)
#         score = np.count_nonzero(detect) / np.count_nonzero(perfect_pattern)
#         if score >= .95:
#             detect_result[j, i] = road_type['RED']

# green_detect = cv2.inRange(hsv_img[:,:,0], 40, 100)
# cv2.imshow('green', green_detect)
# for i in range(7):
#     for j in range(7):
#         if detect_result[j, i] != -1:
#             continue
#         detect = get_block(green_detect, i, j)
#         score = np.count_nonzero(detect) / np.count_nonzero(perfect_pattern)
#         if score >= .95:
#             detect_result[j, i] = road_type['GREEN']

# black_detect =  cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([255, 255, 60])) | cv2.inRange(roi_img, np.array([0, 0, 0]), np.array([50, 50, 50]))
# cv2.imshow('black', black_detect)

# for i in range(7):
#     for j in range(7):
#         if detect_result[j, i] != -1:
#             continue
#         detect = get_block(black_detect, i, j)
#         black_score = np.count_nonzero(detect) / np.count_nonzero(perfect_pattern)
#         if black_score < 0.1:
#             continue

#         SCAN_HEIGHT = 33
#         SCAN_HALF_WIDTH = int(detect.shape[1] / 2)
#         path_score = np.array([0, 0, 0, 0], dtype=np.float32)
#         # UP
#         START_X = int(detect.shape[1] / 2) - SCAN_HALF_WIDTH
#         START_Y = 0
#         UP = detect[START_Y :  START_Y + SCAN_HEIGHT, START_X : START_X + 2 * SCAN_HALF_WIDTH]
#         P_UP = perfect_pattern[START_Y :  START_Y + SCAN_HEIGHT, START_X : START_X + 2 * SCAN_HALF_WIDTH]
#         path_score[0] = np.count_nonzero(UP) / np.count_nonzero(P_UP)
#         # RIGHT
#         START_X = detect.shape[1] - SCAN_HEIGHT
#         START_Y = int(detect.shape[0] / 2) - SCAN_HALF_WIDTH
#         RIGHT = detect[START_Y :  START_Y + 2 * SCAN_HALF_WIDTH, START_X : START_X + SCAN_HEIGHT]
#         P_RIGHT = perfect_pattern[START_Y :  START_Y + 2 * SCAN_HALF_WIDTH, START_X : START_X + SCAN_HEIGHT]
#         path_score[1] = np.count_nonzero(RIGHT) / np.count_nonzero(P_RIGHT)
#         # DOWN
#         START_X = int(detect.shape[1] / 2) - SCAN_HALF_WIDTH
#         START_Y = detect.shape[1] - SCAN_HEIGHT
#         DOWN = detect[START_Y :  START_Y + SCAN_HEIGHT, START_X : START_X + 2 * SCAN_HALF_WIDTH]
#         P_DOWN = perfect_pattern[START_Y :  START_Y + SCAN_HEIGHT, START_X : START_X + 2 * SCAN_HALF_WIDTH]
#         path_score[2] = np.count_nonzero(DOWN) / np.count_nonzero(P_DOWN)
#         # LEFT
#         START_X = 0
#         START_Y = int(detect.shape[0] / 2) - SCAN_HALF_WIDTH
#         LEFT = detect[START_Y :  START_Y + 2 * SCAN_HALF_WIDTH, START_X : START_X + SCAN_HEIGHT]
#         P_LEFT = perfect_pattern[START_Y :  START_Y + 2 * SCAN_HALF_WIDTH, START_X : START_X + SCAN_HEIGHT]
#         path_score[3] = np.count_nonzero(LEFT) / np.count_nonzero(P_LEFT)

#         road = get_road(path_score >= (black_score / 4))
#         if road == road_type['UNPROCESS']:
#             reassign = np.array([False,False,False,False])
#             reassign[path_score.argsort()[:2]] = True
#             road = get_road(reassign)
        
#         detect_result[j, i] = road

# # BLUE
# for i in range(7):
#     for j in range(7):
#         if detect_result[j, i] != -1:
#             continue
#         detect_result[j, i] = road_type['BLUE']


# project_on_board(detect_result, board)
# cv2.imshow("Result",board)
cv2.waitKey(0)
cv2.destroyAllWindows()

