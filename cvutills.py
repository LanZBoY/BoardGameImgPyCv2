import cv2
import os
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

def add_trace_img(name : str, img : np.ndarray, TRACE_MODE = False, trace_imgs : dict[str, np.ndarray] = None):
    if not TRACE_MODE:
        return
    trace_imgs[name] = img

def capture_roi(img : np.ndarray, ROI_SIDE_LENGTH = 600, version : str = 'penguin'):
    start_pt = (int((img.shape[1] - ROI_SIDE_LENGTH) / 2), int((img.shape[0] - ROI_SIDE_LENGTH) / 2 ))
    end_pt = (int((img.shape[1] + ROI_SIDE_LENGTH) / 2), int((img.shape[1] + ROI_SIDE_LENGTH) / 2))
    copy_img = img[start_pt[1] - 10 : end_pt[1] + 10, start_pt[0] - 10 : end_pt[0] + 10, :].copy()
    process_img = cv2.medianBlur(copy_img, 7)
    if version == 'penguin':
        process_img : np.ndarray = cv2.cvtColor(process_img, cv2.COLOR_BGR2HSV_FULL)
    process_img = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
    process_img = cv2.medianBlur(process_img, 7)
    _, process_img = cv2.threshold(process_img, 0., 255., cv2.THRESH_OTSU)
    process_img = cv2.Canny(process_img, 100, 200)
    contours, _ = cv2.findContours(process_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #(x, y)
    contour_img = np.zeros(shape=(process_img.shape[0], process_img.shape[1]), dtype = np.uint8)
    maxidx = getMaxContourIndex(contours)
    cv2.drawContours(contour_img, contours = contours, contourIdx = maxidx, color=255, thickness=1)
    max_contour : np.ndarray = contours[maxidx]
    if max_contour.shape[1] == 1:
        max_contour = max_contour.squeeze(1)
    corners : np.ndarray = cv2.goodFeaturesToTrack(contour_img, 30, 0.1, int(ROI_SIDE_LENGTH / 9), blockSize = 3)
    if corners.shape[1] == 1:
        corners = corners.squeeze(1)
    corners = corners.astype(np.int32)
    selected_pts = getBetterChessCorners(corners=corners)
    # for pt in selected_pts:
    #     cv2.circle(copy_img, pt, 3, (3, 219, 252), thickness = -1)
    if len(selected_pts) != 0:
        project_point = np.array([[0, 0],[723, 0],[0, 723],[723, 723]])
        M = cv2.getPerspectiveTransform(selected_pts.astype(np.float32), project_point.astype(np.float32))
        result_img = cv2.warpPerspective(copy_img, M, (723, 723))
    return result_img

def capture_roi_frame(TRACE_MODE = False, SAVE_PATH = './procedure_img', ROI_SIDE_LENGTH = 600, res = (1280, 720), version : str = 'penguin'):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'Res = {(width, height)}')
    start_pt = (int((width - ROI_SIDE_LENGTH) / 2), int((height - ROI_SIDE_LENGTH) / 2 ))
    end_pt = (int((width + ROI_SIDE_LENGTH) / 2), int((height + ROI_SIDE_LENGTH) / 2))

    goodTracked = False
    detectSignal = False
    trace_imgs = {}
    while(True):
        _, frame = cap.read()
        roi_frame = frame.copy()
        cv2.rectangle(roi_frame, start_pt , end_pt , color=(0, 0, 255))
        if not goodTracked and detectSignal:
            add_trace_img("input", frame, TRACE_MODE, trace_imgs)
            crop_frame = frame[start_pt[1] - 10 : end_pt[1] + 10, start_pt[0] - 10 : end_pt[0] + 10, :].copy()
            add_trace_img("crop", crop_frame, TRACE_MODE, trace_imgs)
            processed_frame = cv2.medianBlur(crop_frame, 7)
            add_trace_img("blur", processed_frame, TRACE_MODE, trace_imgs)
            if version == 'penguin':
                processed_frame : np.ndarray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV_FULL)
                add_trace_img("hsv", processed_frame, TRACE_MODE, trace_imgs)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            add_trace_img("hsv_gray", processed_frame, TRACE_MODE, trace_imgs)
            processed_frame = cv2.medianBlur(processed_frame, 7)
            add_trace_img("hsv_gray_median", processed_frame, TRACE_MODE, trace_imgs)
            # cv2.imshow('hsv_gray_frame', hsv_gray_frame)
            _, processed_frame = cv2.threshold(processed_frame, 0., 255., cv2.THRESH_OTSU)
            add_trace_img("gray_otsu", processed_frame, TRACE_MODE, trace_imgs)
            # cv2.imshow("gray_otsu_frame", processed_frame)
            processed_frame = cv2.Canny(processed_frame, 100, 200)
            add_trace_img("canny", processed_frame, TRACE_MODE, trace_imgs)
            # cv2.imshow("Canny", processed_frame)
            contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #(x, y)
            contour_frame = np.zeros(shape=(processed_frame.shape[0], processed_frame.shape[1]), dtype = np.uint8)
            maxidx = getMaxContourIndex(contours)
            cv2.drawContours(contour_frame, contours = contours, contourIdx = maxidx, color=255, thickness=1)
            max_contour : np.ndarray = contours[maxidx]
            if max_contour.shape[1] == 1:
                max_contour = max_contour.squeeze(1)
            add_trace_img("contour", contour_frame, TRACE_MODE, trace_imgs)
            corners : np.ndarray = cv2.goodFeaturesToTrack(contour_frame, 30, 0.1, int(ROI_SIDE_LENGTH / 9), blockSize = 3)
            corners = corners.astype(np.int32)
            if corners.shape[1] == 1:
                corners = corners.squeeze(1)
            corner_frame = crop_frame.copy()
            for pt in corners:
                cv2.circle(corner_frame, pt, 3, (3, 219, 252), thickness = -1)
            # cv2.imshow("corner_frame", corner_frame)
            add_trace_img("corner", corner_frame, TRACE_MODE, trace_imgs)
            selected_pts = getBetterChessCorners(corners=corners)
            if len(selected_pts) != 0:
                select_pic = crop_frame.copy()
                for pt in selected_pts:
                    cv2.circle(select_pic, pt, 3, (3, 219, 252), thickness = -1)
                add_trace_img('select', select_pic, TRACE_MODE, trace_imgs)
                # 開始映射轉換
                project_point = np.array([[0, 0],[728, 0],[0, 728],[728, 728]])
                M = cv2.getPerspectiveTransform(selected_pts.astype(np.float32), project_point.astype(np.float32))
                result_img = cv2.warpPerspective(crop_frame, M, (736, 736))
                add_trace_img("result", result_img, TRACE_MODE, trace_imgs)
                goodTracked = True
        
        if detectSignal:
            cv2.putText(roi_frame, "Detecting...", org=(0, 25), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness = 3)
        else:
            cv2.putText(roi_frame, "Press D to Detect", org=(0, 25), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness = 3)

        cv2.imshow("ROI", roi_frame)
        # 

        if cv2.waitKey(1) & 0xFF == ord('d'):
            detectSignal = not detectSignal
            goodTracked = False

        if goodTracked:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    
    return result_img, trace_imgs
