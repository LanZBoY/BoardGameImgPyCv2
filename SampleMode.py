import cv2
import os 
import numpy as np
from cvutills import *
# BLOCK_SIZE = 100

VERSION = "penguin" # penguin | space

SAVE_PATH = '.\\sample'
SAVE_NAME = 'sample'
TRACE_MODE = True
ROI_SIDE_LENGTH = 600

for i in range(10):
    roi_img, trace_imgs = capture_roi_frame(TRACE_MODE, SAVE_PATH, ROI_SIDE_LENGTH, (1280, 720), version=VERSION)
    cv2.destroyAllWindows()
    img_num = len(os.listdir(SAVE_PATH))
    cv2.imwrite(os.path.join(SAVE_PATH, f'{SAVE_NAME}_{img_num}.jpg'), trace_imgs['input'])
    cv2.imshow(f'sample', trace_imgs['input'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# cv2.imwrite('sample.jpg', trace_imgs['input'])
cv2.destroyAllWindows()