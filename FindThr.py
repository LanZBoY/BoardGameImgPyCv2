
import numpy as np
import cv2

min_0_thr = 0
min_1_thr = 0
min_2_thr = 0
max_0_thr = 255
max_1_thr = 255
max_2_thr = 255

R = 0
G = 0
B = 0

src = cv2.imread("crop.jpg")
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
src = cv2.cvtColor(src, cv2.COLOR_RGB2HSV_FULL)
result = np.array([])

# def change_R(val):
#     global R
#     R = val
#     onRGBChange()

# def change_G(val):
#     global G
#     G = val
#     onRGBChange()

# def change_B(val):
#     global B
#     B = val
#     onRGBChange()

def onRGBChange():
    global B, G, R
    draw = np.zeros(shape=(400, 400, 3), dtype=np.uint8)
    draw[:, :, 0].fill(B)
    draw[:, :, 1].fill(G)
    draw[:, :, 2].fill(R)
    cv2.imshow("draw", draw)

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
    global src, min_0_thr, min_1_thr, min_2_thr, max_0_thr, max_1_thr, max_2_thr, result
    result = cv2.inRange(src, np.array([min_0_thr, min_1_thr, min_2_thr]), np.array([max_0_thr, max_1_thr, max_2_thr]))
    cv2.imshow("result", result)


cv2.namedWindow("ControlPannel")
cv2.resizeWindow("ControlPannel", 700, 300)
cv2.createTrackbar('min_0_thr','ControlPannel', 0, 255, change_min_0)
cv2.createTrackbar('max_0_thr','ControlPannel', 0, 255, change_max_0)
cv2.createTrackbar('min_1_thr','ControlPannel', 0, 255, change_min_1)
cv2.createTrackbar('max_1_thr','ControlPannel', 0, 255, change_max_1)
cv2.createTrackbar('min_2_thr','ControlPannel', 0, 255, change_min_2)
cv2.createTrackbar('max_2_thr','ControlPannel', 0, 255, change_max_2)
# cv2.createTrackbar('R','Pannel', 0, 255, change_R)
# cv2.createTrackbar('G','Pannel', 0, 255, change_G)
# cv2.createTrackbar('B','Pannel', 0, 255, change_B)
cv2.waitKey(0)
# cv2.imwrite("cool.jpg", result)
# RGB BLACK lowB = [0,0,0], highB = [75,75,75]
# RGB BLUE lowB = [0,0,70], highB = [80,120,255]
cv2.destroyAllWindows()