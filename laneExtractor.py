import numpy as np
from numpy import linalg
from cv2 import cv2


def getLanes(image_np):
    img_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HLS)
    ysize = img_gray.shape[0]
    xsize = img_gray.shape[1]

    # Detecting yellow and white colors
    low_yellow = np.array([20, 100, 100])
    high_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, low_yellow, high_yellow)
    mask_white = cv2.inRange(img_gray, 150, 255)

    mask_yw = cv2.bitwise_or(mask_yellow, mask_white)
    mask_onimage = cv2.bitwise_and(img_gray, mask_yw)

    # Smoothing for removing noise
    gray_blur = cv2.GaussianBlur(mask_onimage, (5, 5), 0)

    # Region of Interest Extraction
    mask_roi = np.zeros(img_gray.shape, dtype=np.uint8)
    left_bottom = [0, ysize]
    right_bottom = [xsize-0, ysize]
    apex_left = [0, ((ysize/2)+50)]
    apex_right = [xsize-0, ((ysize/2)+50)]
    mask_color = 255
    roi_corners = np.array(
        [[left_bottom, apex_left, apex_right, right_bottom]], dtype=np.int32)
    cv2.fillPoly(mask_roi, roi_corners, mask_color)
    image_roi = cv2.bitwise_and(gray_blur, mask_roi)

    # Thresholding before edge
    ret, img_postthresh = cv2.threshold(
        image_roi, 50, 255, cv2.THRESH_BINARY)

    # Use canny edge detection
    edge_low = 50
    edge_high = 200
    img_edge = cv2.Canny(img_postthresh, edge_low, edge_high)

    # Hough Line Draw
    road_lines = cv2.HoughLinesP(img_edge, 1, np.pi/180, 20)
    left_lane, right_lane = extract_lane(road_lines)
    lanes = split_append(left_lane, right_lane, xsize, ysize)
    return lanes


def extract_lane(road_lines):
    left_lane = []
    right_lane = []

    if road_lines is not None:
        for x in range(0, len(road_lines)):
            for x1, y1, x2, y2 in road_lines[x]:
                slope = compute_slope(x1, y1, x2, y2)
                if slope is None:
                    continue
                if (slope < 0):
                    line = road_lines[x]
                    left_lane.append(road_lines[x])
                else:
                    if (slope >= 0):
                        line = road_lines[x]
                        right_lane.append(road_lines[x])

    return left_lane, right_lane


def compute_slope(x1, y1, x2, y2):
    if x2 != x1:
        return ((y2-y1)/(x2-x1))


def getLanesList(lanes, xsize, ysize, laneType):
    lanes = sorted(lanes, key=lambda x : (x[0][1],x[0][0]))
    lanesList = []
    for x in range(0, len(lanes)):
        for x1, y1, x2, y2 in lanes[x]:
            if laneType == 'left' and (x1 >= (xsize/2) or x2 >= (xsize/2)):
                continue
            if laneType == 'right' and (x1 <= (xsize/2) or x2 <= (xsize/2)):
                continue
            appended = False
            for index in range(0, len(lanesList)):
                
                p1 = np.array(lanesList[index][0])
                p2 = np.array(lanesList[index][1])
                p3 = np.array([x1, y1])
                p4 = np.array([x2, y2])
                d1 = np.abs(np.cross(p2-p1, p1-p3)) / linalg.norm(p2-p1)
                d2 = np.abs(np.cross(p2-p1, p1-p4)) / linalg.norm(p2-p1)
                if d1 < 45 or d2 < 45:
                    lanesList[index].append([x1, y1])
                    lanesList[index].append([x2, y2])
                    appended = True
            if not appended:
                lane = []
                lane.append([x1, y1])
                lane.append([x2, y2])
                lanesList.append(lane)
    return lanesList


def split_append(left_lanes, right_lanes, xsize, ysize):
    leftlanesList = getLanesList(left_lanes, xsize, ysize, 'left')
    rightlanesList = getLanesList(right_lanes, xsize, ysize,'right')

    lanesList = []
    for x in range(0, len(leftlanesList)):
        laneTmp = np.array(leftlanesList[x])
        laneTmp = laneTmp[np.argsort(laneTmp[:, 0])]
        lanesList.append(laneTmp)
    for x in range(0, len(rightlanesList)):
        laneTmp = np.array(rightlanesList[x])
        laneTmp = laneTmp[np.argsort(laneTmp[:, 0])]
        lanesList.append(laneTmp)
    return lanesList