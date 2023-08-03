import cv2
import numpy as np
from cvutills import *

root = './procedure_img'
ROI_SIDE_LENGTH = 600
RESOLUTION = (1280, 720)
N_FRAME = 1 # 每N禎 偵測一次

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f'Res = {(width, height)}')

start_pt = (int((width - ROI_SIDE_LENGTH) / 2), int((height - ROI_SIDE_LENGTH) / 2 ))
end_pt = (int((width + ROI_SIDE_LENGTH) / 2), int((height + ROI_SIDE_LENGTH) / 2))

goodTracked = False
detectSignal = False
current_frame = 0

while(True):
    _, frame = cap.read()
    roi_frame = frame.copy()
    
    cv2.rectangle(roi_frame, start_pt , end_pt , color=(0, 0, 255))
    if (not goodTracked) and detectSignal and (current_frame == 0):
        crop_frame = frame[start_pt[1] - 10 : end_pt[1] + 10, start_pt[0] - 10 : end_pt[0] + 10, :].copy()
        blur_frame = cv2.medianBlur(crop_frame, 7)
        # cv2.imshow('blur', blur_frame)
        hsv_frame : np.ndarray = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV_FULL)
        hsv_gray_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2GRAY)
        hsv_gray_frame = cv2.medianBlur(hsv_gray_frame, 7)
        # cv2.imshow('hsv_gray_frame', hsv_gray_frame)
        _, gray_otsu_frame = cv2.threshold(hsv_gray_frame, 0., 255., cv2.THRESH_OTSU)
        cv2.imshow("gray_otsu_frame", gray_otsu_frame)
        canny_frame = cv2.Canny(gray_otsu_frame, 100, 200)
        cv2.imshow("Canny", canny_frame)
        contours, _ = cv2.findContours(canny_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #(x, y)
        contour_frame = np.zeros(shape=(blur_frame.shape[0], blur_frame.shape[1]), dtype = np.uint8)
        maxidx = getMaxContourIndex(contours)
        cv2.drawContours(contour_frame, contours = contours, contourIdx = maxidx, color=255, thickness=1)
        max_contour : np.ndarray = contours[maxidx]
        if max_contour.shape[1] == 1:
            max_contour = max_contour.squeeze(1)
        cv2.imshow("contour_frame", contour_frame)

        corners : np.ndarray = cv2.goodFeaturesToTrack(contour_frame, 30, 0.1, int(ROI_SIDE_LENGTH / 9), blockSize = 3)
        corners = corners.astype(np.int32)
        if corners.shape[1] == 1:
            corners = corners.squeeze(1)
        # corner_frame = crop_frame.copy()
        # for pt in corners:
        #     cv2.circle(corner_frame, pt, 3, (3, 219, 252), thickness = -1)
        # cv2.imshow("corner_frame", corner_frame)
        selected_pts = getBetterChessCorners(corners=corners)
        if len(selected_pts) != 0:
            # select_pic = crop_frame.copy()
            # for pt in selected_pts:
            #     cv2.circle(select_pic, pt, 3, (3, 219, 252), thickness = -1)
            # 開始映射轉換
            project_point = np.array([[0, 0],[736, 0],[0, 736],[736, 736]])
            M = cv2.getPerspectiveTransform(selected_pts.astype(np.float32), project_point.astype(np.float32))
            result_img = cv2.warpPerspective(crop_frame, M, (736, 736))
            goodTracked = True
    
    if detectSignal:
        cv2.putText(roi_frame, "Detecting...", org=(0, 25), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness = 3)
    else:
        cv2.putText(roi_frame, "Press D to Detect", org=(0, 25), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness = 3)
    cv2.imshow("ROI", roi_frame)

    # 偵測紐
    if cv2.waitKey(1) & 0xFF == ord('d'):
        detectSignal = not detectSignal
        goodTracked = False

    if goodTracked and detectSignal:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    current_frame = (current_frame + 1) % N_FRAME

cap.release()
cv2.destroyAllWindows()

if goodTracked:
    cv2.imshow("final", result_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

