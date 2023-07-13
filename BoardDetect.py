import cv2
import numpy as np
from cvutills import *

cap = cv2.VideoCapture(0)
ROI_SIDE_LENGTH = 400
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Res = {(width, height)}')
start_pt = (int((width - ROI_SIDE_LENGTH) / 2), int((height - ROI_SIDE_LENGTH) / 2 ))
end_pt = (int((width + ROI_SIDE_LENGTH) / 2), int((height + ROI_SIDE_LENGTH) / 2))
goodTracked = False
while(True):
    _, frame = cap.read()
    roi_frame = frame.copy()
    cv2.rectangle(roi_frame, start_pt , end_pt , color=(0, 0, 255))
    cv2.imshow("ROI", roi_frame)
    if not goodTracked:
        crop_frame = frame[start_pt[1] - 10 : end_pt[1] + 10, start_pt[0] - 10 : end_pt[0] + 10, :].copy()
        blur_frame = cv2.medianBlur(crop_frame, 7)
        # cv2.imshow('blur', blur_frame)
        hsv_frame : np.ndarray = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV_FULL)
        hsv_gray_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2GRAY)
        hsv_gray_frame = cv2.medianBlur(hsv_gray_frame, 7)
        # cv2.imshow('hsv_gray_frame', hsv_gray_frame)
        _, gray_otsu_frame = cv2.threshold(hsv_gray_frame, 0., 255., cv2.THRESH_OTSU)
        # cv2.imshow("gray_otsu_frame", gray_otsu_frame)
        canny_frame = cv2.Canny(gray_otsu_frame, 100, 200)
        # cv2.imshow("Canny", canny_frame)
        contours, _ = cv2.findContours(canny_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #(x, y)
        contour_frame = np.zeros(shape=(blur_frame.shape[0], blur_frame.shape[1]), dtype = np.uint8)
        maxidx = getMaxContourIndex(contours)
        cv2.drawContours(contour_frame, contours = contours, contourIdx = maxidx, color=255, thickness=1)
        max_contour : np.ndarray = contours[maxidx]
        if max_contour.shape[1] == 1:
            max_contour = max_contour.squeeze(1)
        # cv2.imshow("contour_frame", contour_frame)

        corners : np.ndarray = cv2.goodFeaturesToTrack(contour_frame, 30, 0.1, int(ROI_SIDE_LENGTH / 9), blockSize = 3)
        corners = corners.astype(np.int32)
        if corners.shape[1] == 1:
            corners = corners.squeeze(1)
        corner_frame = crop_frame.copy()
        for pt in corners:
            cv2.circle(corner_frame, pt, 3, (3, 219, 252), thickness = -1)
        # cv2.imshow("corner_frame", corner_frame)
        selected_pts = getBetterChessCorners(corners=corners)
        if len(selected_pts) != 0:
            select_pic = crop_frame.copy()
            for pt in selected_pts:
                cv2.circle(select_pic, pt, 3, (3, 219, 252), thickness = -1)
            # 開始映射轉換
            project_point = np.array([[0, 0],[736, 0],[0, 736],[736, 736]])
            M = cv2.getPerspectiveTransform(selected_pts.astype(np.float32), project_point.astype(np.float32))
            result_img = cv2.warpPerspective(crop_frame, M, (736, 736))
            if not goodTracked:
                goodTracked = True
                print("OK")
    # 
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if goodTracked:
            cv2.imshow("select_pic", select_pic)
            cv2.imshow("final", result_img)
            goodTracked = False
        else:
            print("請重新偵測")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

