import cv2
import os 
import numpy as np
from cvutills import *

# road_type = {
#         "UNKNOWN" : -3,
#         "UN_PROCESS" : -2,
#         "BLANK" : -1,
#         "RED" : 0,
#         "GREEN" : 1,
#         "BLUE" : 2,
#         "BLACK" : 3,
#         "B_STRAIGHT_HORIZON" : 4,
#         "B_STRAIGHT_VERTICAL" : 5,
#         "B_UP_RIGHT" : 6,
#         "B_LEFT_DOWN" : 7,
#         "B_LEFT_UP" : 8,
#         "B_DOWN_RIGHT" : 9
#     }



SAVE_PATH = './procedure_img'
TRACE_MODE = False
ROI_SIDE_LENGTH = 600

img, trace_imgs = capture_roi_frame(TRACE_MODE, SAVE_PATH, ROI_SIDE_LENGTH, (1280, 720))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray_img)
_, otsu_img = cv2.threshold(gray_img, 0., 255., cv2.THRESH_OTSU)
cv2.imshow('otsu', otsu_img)
for step, key in enumerate(trace_imgs.keys()):
    cv2.imwrite(os.path.join(SAVE_PATH, f'{step}_{key}.png'), trace_imgs[key])
    print(f'{step}_{key}.png')

cv2.waitKey(0)
cv2.destroyAllWindows()

