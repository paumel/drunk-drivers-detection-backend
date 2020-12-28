import numpy as np
from numpy import linalg
from cv2 import cv2


def getLanes(image_np, output_dict, MIN_SCORE, height, width):
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
    road_lines = cv2.HoughLinesP(
        img_edge, 
        rho=6,
        theta=np.pi / 180,
        threshold=100,
        lines=np.array([]),
        minLineLength=20,
        maxLineGap=100
        )

    # if road_lines is not None:
    #     for i in range(0, len(road_lines)):
    #         l = road_lines[i][0]
    #         cv2.line(image_np, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    # cv2.imshow('image_testas', image_np)
    # cv2.imshow('image_test2', img_edge)

    indexesToDelete = []
    if road_lines is not None:
        for x in range(0, len(road_lines)):
            for x1, y1, x2, y2 in road_lines[x]:
                boxesIndex = 0
                while output_dict['detection_scores'][boxesIndex] > MIN_SCORE:
                    box = output_dict['detection_boxes'][boxesIndex]
                    (ymin, xmin, ymax, xmax) = (box[0]*height, box[1]*width, box[2]*height, box[3]*width)
                    if (xmin <= x1 and x1 <= xmax and ymin <= y1 and y1 <= ymax) or (xmin <= x2 and x2 <= xmax and ymin <= y2 and y2 <= ymax):
                        indexesToDelete.append(x)
                        break
                    boxesIndex += 1

    indexesToDelete.sort(reverse=True)
    for indexToDelete in indexesToDelete:
        road_lines = np.delete(road_lines, indexToDelete, axis=0)

    left_lane, right_lane, left_slopes, right_slopes = extract_lane(road_lines)
    lanes = split_append(left_lane, right_lane, xsize, ysize, left_slopes, right_slopes, image_np)
    return lanes


def extract_lane(road_lines):
    left_lane = []
    right_lane = []
    left_slopes = []
    right_slopes = []

    if road_lines is not None:
        for x in range(0, len(road_lines)):
            for x1, y1, x2, y2 in road_lines[x]:
                slope = compute_slope(x1, y1, x2, y2)
                if slope is None:
                    continue
                if (slope < 0):
                    line = road_lines[x]
                    left_lane.append(road_lines[x])
                    left_slopes.append(slope)
                else:
                    if (slope >= 0):
                        line = road_lines[x]
                        right_lane.append(road_lines[x])
                        right_slopes.append(slope)
    left_slopes = cluster(left_slopes, 0.1)
    right_slopes = cluster(right_slopes, 0.1)
    return left_lane, right_lane, left_slopes, right_slopes


def cluster(data, maxgap):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*

        >>> cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)
        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

        >>> cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)
        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

    '''
    if len(data) == 0:
        return []
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def compute_slope(x1, y1, x2, y2):
    if x2 != x1:
        return ((y2-y1)/(x2-x1))


def getLanesList(lanes, xsize, ysize, laneType, slopes, image_np):
    lanes = sorted(lanes, key=lambda x : (x[0][1],x[0][0]))
    # if laneType == 'left':
    #     if lanes is not None:
    #         for i in range(0, len(lanes)):
    #             l = lanes[i][0]
    #             cv2.line(image_np, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    # if laneType == 'right':
    #     if lanes is not None:
    #         for i in range(0, len(lanes)):
    #             l = lanes[i][0]
    #             cv2.line(image_np, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
    # cv2.imshow('image_test3', image_np)
    lanesList = []
    for slope in slopes:
        lanesList.append([])
    for x in range(0, len(lanes)):
        for x1, y1, x2, y2 in lanes[x]:
            slope = compute_slope(x1, y1, x2, y2)
            x = [x for x in slopes if slope in x][0]
            index = slopes.index(x)
            lanesList[index].append([x1, y1])
            lanesList[index].append([x2, y2])
    
    
    # for x in range(0, len(lanes)):
    #     for x1, y1, x2, y2 in lanes[x]:
    #         if laneType == 'left' and (x1 >= (xsize/2) or x2 >= (xsize/2)):
    #             continue
    #         if laneType == 'right' and (x1 <= (xsize/2) or x2 <= (xsize/2)):
    #             continue
    #         appended = False
    #         for index in range(0, len(lanesList)):
                
    #             p1 = np.array(lanesList[index][0])
    #             p2 = np.array(lanesList[index][1])
    #             p3 = np.array([x1, y1])
    #             p4 = np.array([x2, y2])
    #             d1 = np.abs(np.cross(p2-p1, p1-p3)) / linalg.norm(p2-p1)
    #             d2 = np.abs(np.cross(p2-p1, p1-p4)) / linalg.norm(p2-p1)
    #             if d1 < 30 or d2 < 30:
    #                 lanesList[index].append([x1, y1])
    #                 lanesList[index].append([x2, y2])
    #                 appended = True
    #         if not appended:
    #             lane = []
    #             lane.append([x1, y1])
    #             lane.append([x2, y2])
    #             lanesList.append(lane)
    if laneType == 'left':
        colors = [
            (255,0,0),
            (255,255,0),
            (255,0,255),
            (0,255,0),
            (0,255,255),
            (0,0,255),
            (255,255,255),
        ]
    else:
        colors = [
            (255,255,255),
            (0,0,255),
            (0,255,255),
            (0,255,0),
            (255,0,255),
            (255,255,0),
            (255,0,0),
        ]
    # if lanesList is not None:
    #     for j in range(0, len(lanesList)):
    #         i = 0
    #         while i < len(lanesList[j]):
    #             l = lanesList[j][i]
    #             l2 = lanesList[j][i+1]
    #             cv2.line(image_np, (l[0], l[1]), (l2[0], l2[1]), colors[j], 3, cv2.LINE_AA)
    #             i = i + 2
    # cv2.imshow('image_test', image_np)
    
    return lanesList


def split_append(left_lanes, right_lanes, xsize, ysize, left_slopes, right_slopes, image_np):
    leftlanesList = getLanesList(left_lanes, xsize, ysize, 'left', left_slopes, image_np)
    rightlanesList = getLanesList(right_lanes, xsize, ysize,'right', right_slopes, image_np)

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