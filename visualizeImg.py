import carDetector
import numpy as np
from cv2 import cv2
from object_detection.utils import visualization_utils as vis_util
from sklearn import linear_model
import intersect

def visualizeCars(image_np, output_dict, MIN_SCORE):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        carDetector.getCategoryIndex(),
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh=MIN_SCORE,
        line_thickness=5)


def visualizeCarTrajectory(image_np, currentFrame, currentFrameLines, framePoints, frameLines, framesIndex, allLines):
    for i in range(len(currentFrame)):
        color = tuple(np.random.randint(256, size=3))
        color = (int(color[0]), int(color[1]), int(color[2]))
        # cv2.line(image_np, (
        #     int(round(currentFrameLines[i][0])),
        #     int(round(currentFrameLines[i][2]))
        # ), (
        #     int(round(currentFrameLines[i][1])),
        #     int(round(currentFrameLines[i][2]))
        # ), (0, 255, 255), 3)

        lastPoints, framesIndexes = carDetector.lastNumberOfPoints(
            currentFrame[i], framePoints, framesIndex, 100)

        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for indexes in framesIndexes:
            x1.append(frameLines[indexes[0]][indexes[1]][0])
            y1.append(frameLines[indexes[0]][indexes[1]][2])
            x2.append(int(round(frameLines[indexes[0]][indexes[1]][1])))
            y2.append(int(round(frameLines[indexes[0]][indexes[1]][2])))

        drawPolyline(x1, y1, image_np, color, allLines, currentFrameLines[i])
        drawPolyline(x2, y2, image_np, color, allLines, currentFrameLines[i])


def drawPolyline(x, y, img, color, allLines, currentCar):
    if len(x) > 1 and len(y) > 1:
        try:
            z = np.polyfit(x, y, 2)
            lspace = np.linspace(min(x), max(x), 100)
            draw_x = lspace
            draw_y = np.polyval(z, draw_x)

            for line in allLines:
                line_x = [line[0][0], line[1][0]]
                line_y = [line[0][1], line[1][1]]
                x,y = intersect.intersection(draw_x, draw_y, line_x, line_y)
                for i, x_coord in enumerate(x):
                    cv2.circle(
                        img, (
                            int(round(x_coord)),
                            int(round(y[i]))
                        ),
                        7,
                        (0, 0, 255),
                        -1
                    )
                if len(x) >= 1:
                    cv2.line(img, (
                        int(round(currentCar[0])),
                        int(round(currentCar[2]))
                    ), (
                        int(round(currentCar[1])),
                        int(round(currentCar[2]))
                    ), (0, 0, 255), 3)


            draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
            cv2.polylines(img, [draw_points], False, tuple(color), thickness=3)
        except ValueError:
            print('All zero values in polyfit')


def ransac_drawlane(lanes, frame):
    allLines = []
    for lane_sa in lanes:
        if lane_sa.size <= 4:
            continue
        lane_x = []
        lane_y = []

        for x1, y1 in lane_sa:
            lane_x.append([x1])
            lane_y.append([y1])

        ransac_x = np.array(lane_x)
        ransac_y = np.array(lane_y)

        ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        try:
            ransac.fit(ransac_x, ransac_y)
        except ValueError:
            print("ransac error")
            continue
        slope = ransac.estimator_.coef_
        if slope[0][0] < 0.2 and slope[0][0] > -0.2:
            continue
        intercept = ransac.estimator_.intercept_

        ysize = frame.shape[0]
        xsize = frame.shape[1]
        y_limit_low = int(0.95*ysize)
        y_limit_high = int(0.55*ysize)

        # Coordinates for point 1(Bottom Left)
        y_1 = ysize
        x_1_float = (y_1-intercept)/slope
        if x_1_float[0][0] == float("inf") or x_1_float[0][0] == float("-inf"):
            continue
        x_1 = int(x_1_float)

        # Coordinates for point 2(Bottom Left)
        y_2 = y_limit_high
        x_2_float = (y_2-intercept)/slope
        if x_2_float[0][0] == float("inf") or x_2_float[0][0] == float("-inf"):
            continue
        x_2 = int(x_2_float)

        cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 255), 3)
        allLines.append([[x_1, y_1], [x_2, y_2]])
    mask_color = (255, 255, 0)
    frame_copy = frame.copy()
    opacity = 0.4
    cv2.addWeighted(frame_copy, opacity, frame, 1-opacity, 0, frame)
    return frame, allLines