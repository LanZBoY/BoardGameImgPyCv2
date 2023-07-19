import cv2
import os 
import numpy as np
from cvutills import *

SAVE_PATH = './procedure_img'
TRACE_MODE = True
ROI_SIDE_LENGTH = 600

img, trace_imgs = capture_roi(TRACE_MODE, SAVE_PATH, ROI_SIDE_LENGTH, (1280, 720))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray_img)
_, otsu_img = cv2.threshold(gray_img, 0., 255., cv2.THRESH_OTSU)
cv2.imshow('otsu', otsu_img)
for step, key in enumerate(trace_imgs.keys()):
    cv2.imwrite(os.path.join(SAVE_PATH, f'{step}_{key}.png'), trace_imgs[key])
    print(f'{step}_{key}.png')

cv2.waitKey(0)
cv2.destroyAllWindows()

