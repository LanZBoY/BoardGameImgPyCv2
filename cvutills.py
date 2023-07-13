import cv2
import numpy as np

def getBetterChessCorners(corners : np.ndarray):
    horizon_select = getChessCorners(corners=corners)
    vertical_select = getChessCorners(corners=corners, vertical_scan=True)
    if len(horizon_select) == 0 or len(vertical_select) == 0:
        return np.array([])
    if np.all(horizon_select == horizon_select[0]) or np.all(vertical_select == vertical_select[0]):
        return np.array([])
    if np.sum(horizon_select - vertical_select) != 0:
        return np.array([])
    return horizon_select

def getChessCorners(corners : np.ndarray, vertical_scan = False) -> np.ndarray:
    if len(corners) == 0:
        return np.array([])
    if not vertical_scan:
        min_y = corners[:, 1].min()
        max_y = corners[:, 1].max()
        bin = int((max_y - min_y) / 8)
        bias = int(bin * 0.25)
        up_pts = corners[(min_y + bin - bias <= corners[:, 1]) & (corners[:, 1] <= min_y + bin + bias)]
        down_pts = corners[(max_y - bias <= corners[:, 1]) & (corners[:, 1] <= max_y)]
        if len(up_pts) == 0 or len(down_pts) == 0:
            return np.array([])
        return np.array([
            up_pts[up_pts[:, 0].argmin()],
            up_pts[up_pts[:, 0].argmax()],
            down_pts[down_pts[:, 0].argmin()],
            down_pts[down_pts[:, 0].argmax()],
        ])
    min_x = corners[:, 0].min()
    max_x = corners[:, 0].max()
    bin = int((max_x - min_x) / 9)
    bias = int(bin * 0.25)
    left_pts = corners[(min_x + bin - bias <= corners[:, 0]) & (corners[:, 0] <= min_x + bin + bias)]
    right_pts = corners[(max_x - bin - bias <= corners[:, 0]) & (corners[:, 0] <= max_x - bin + bias)]

    if len(right_pts) == 0 or len(left_pts) == 0:
        return np.array([])
    
    return np.array(
        [
            left_pts[left_pts[:, 1].argmin()], 
            right_pts[right_pts[:, 1].argmin()], 
            left_pts[left_pts[:, 1].argmax()],
            right_pts[right_pts[:, 1].argmax()],
            ]
    )
    

def getMaxContourIndex(contours):
    maxidx = -1
    max = -1 
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max:
            max = area
            maxidx = i
    return maxidx

def order_points(pts: np.ndarray):
    sort_x = pts[np.argsort(pts[:, 0]), :]
    left = sort_x[:2, :]
    right = sort_x[2:, :]

    left = left[np.argsort(left[:, 1])[::-1], :]
    right = right[np.argsort(right[:, 1]), :]
    return np.concatenate((left, right), axis = 0)

def draw_vertical(img : np.ndarray, x, color = (255, 0, 0), thickness = 2):
    y = img.shape[0]
    reuslt = cv2.line(img, (x, 0), (x, y), color = color, thickness = thickness)
    return reuslt