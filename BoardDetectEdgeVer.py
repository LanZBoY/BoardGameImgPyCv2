import cv2
import numpy as np
from cvutills import *

cap = cv2.VideoCapture(0)
ROI_SIDE_LENGTH = 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Res = {(width, height)}')
start_pt = (int((width - ROI_SIDE_LENGTH) / 2), int((height - ROI_SIDE_LENGTH) / 2 ))
end_pt = (int((width + ROI_SIDE_LENGTH) / 2), int((height + ROI_SIDE_LENGTH) / 2))
while(True):
    _, frame = cap.read()
    roi_frame = frame.copy()
    cv2.rectangle(roi_frame, start_pt , end_pt , color=(0, 0, 255))
    cv2.imshow("ROI", roi_frame)
    crop_frame = frame[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :].copy()
    blur_frame = cv2.medianBlur(crop_frame, 7)
    # cv2.imshow('blur', blur_frame)
    hsv_frame : np.ndarray = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV_FULL)
    hsv_gray_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2GRAY)
    hsv_gray_frame = cv2.medianBlur(hsv_gray_frame, 7)
    _, gray_otsu_frame = cv2.threshold(hsv_gray_frame, 0., 255., cv2.THRESH_OTSU)
    # cv2.imshow("gray_otsu_frame", gray_otsu_frame)
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
    # 
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # (x, y)
        # line_pic = draw_vertical(crop_frame.copy(), min_x + bin, thickness = 1)
        # draw_vertical(line_pic, min_x + bin - bias, color=(0, 255, 0), thickness = 1)
        # draw_vertical(line_pic, min_x + bin + bias, color=(0, 255, 0), thickness = 1)
        # draw_vertical(line_pic, max_x - bin, thickness = 1)
        # draw_vertical(line_pic, max_x - bin - bias, color=(0, 255, 0), thickness = 1)
        # draw_vertical(line_pic, max_x - bin + bias, color=(0, 255, 0), thickness = 1)
        # cv2.imshow("line_pic",line_pic)
        # 最小區間 min_x + bin
        # 最大區間 max_x - bin
        # 找符合區間內的點
        min_x = max_contour[:, 0].min()
        max_x = max_contour[:, 0].max()
        bin = int((max_x - min_x) / 9)
        bias = int(bin * 0.4)
        print(f'(bin, bias) = {(bin, bias)}')
        select_pic = crop_frame.copy()
        right_pt = max_contour[(max_x - bin - bias <= max_contour[:, 0]) & (max_contour[:, 0] <= max_x - bin + bias)]
        left_pt = max_contour[(min_x + bin - bias <= max_contour[:, 0]) & (max_contour[:, 0] <= min_x + bin + bias)]
        if len(right_pt) == 0 or len(left_pt) == 0:
            print("Empty Point")
            continue

        selected_pt = np.array(
            [
                left_pt[left_pt[:, 1].argmin()], 
                right_pt[right_pt[:, 1].argmin()], 
                left_pt[left_pt[:, 1].argmax()]],
                right_pt[right_pt[:, 1].argmax()] 
        )
        for pt in selected_pt:
            cv2.circle(select_pic, pt, 3, (3, 219, 252), thickness = -1)
        cv2.imshow("select_pic", select_pic)
        # 開始映射轉換
        project_point = np.array([[0, 0],[700, 0],[0, 700],[700, 700]])
        M = cv2.getPerspectiveTransform(selected_pt.astype(np.float32), project_point.astype(np.float32))
        result_img = cv2.warpPerspective(crop_frame, M, (700, 700))
        cv2.imshow("final", result_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

