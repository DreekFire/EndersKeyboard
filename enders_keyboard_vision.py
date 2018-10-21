import cv2 as cv
import numpy as np
import imutils
import time
import enders_keyboard
import math

bg = None

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold = 10):
    kernel = np.ones((5,5),np.uint8)

    global bg
    diff = cv.absdiff(bg.astype("uint8"), image)
    thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]
    cv.GaussianBlur(thresholded, (11, 11), 0)
    thresholded = cv.dilate(thresholded, kernel, 10)
    thresh = cv.threshold(thresholded, 100, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, kernel, 20)
    (_, conts, _) = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(conts) == 0:
        return
    else:
        segmented = max(conts, key=cv.contourArea)
        return (thresh, segmented)

def rough_hull(hull_ids, cont, max_dist):
    if len(hull_ids) > 0 and len(cont) > 0:
        res = []
        current_pos = cont[hull_ids[0]][0][0]
        points = []
        for point in hull_ids:
            dist = np.linalg.norm(cont[point][0][0] - current_pos)
            """if dist < max_dist:
                current_pos = [(current_pos[0] * len(points) + cont[point][0][0][1]) / (len(points) + 1), (current_pos[1] * len(points) + cont[point][0][0][1]) / (len(points) + 1)]
                points.append(point)
            else:
                for i in points:
                    chosen = None
                    min = 100000000
                    cur = np.linalg.norm(cont[i][0][0] - current_pos)
                    if cur < min:
                        min = cur
                        chosen = i
                res.append(i)
                points = [point]"""
            if dist > max_dist:
                res.append(point)
                current_pos = cont[point][0][0]
        return res
    else:
        return []

"""def defect_vertices(cont, indices):
    if len(cont[0][0]) > 0 and indices is not None and len(indices) > 0:
        defects = cv.convexityDefects(cont, indices)
    else:
        return []

    neighbors = {}
    if defects is not None:
        for defect in defects:
            dist = dist_to_line([cont[defect[0][2]][0][0], cont[defect[0][2]][0][1]], [[cont[defect[0][0]][0][0], cont[defect[0][0]][0][1]], [cont[defect[0][1]][0][0], cont[defect[0][1]][0][1]]])
            #if dist > 5: #idk how contour distances work
            start_pt_id = defect[0][0]
            end_pt_id = defect[0][1]
            defect_pt_id = defect[0][2]
            if start_pt_id not in neighbors:
                neighbors[start_pt_id] = [defect_pt_id]
            else:
                neighbors[start_pt_id].append(defect_pt_id)
            if end_pt_id not in neighbors:
                neighbors[end_pt_id] = [defect_pt_id]
            else:
                neighbors[end_pt_id].append(defect_pt_id)

    res = []
    for key in neighbors:
        if len(neighbors[key]) > 1:
            res.append([cont[key], cont[neighbors[key][0]], cont[neighbors[key][1]], dist])
    return res

def dist_to_line(point, line):
    return np.abs((line[1][1] - line[0][1]) * point[0] - (line[1][0] - line[0][0]) * point[1] + line[1][0] * line[0][1] - line[1][1] * line[0][0]) / np.sqrt((line[1][1] - line[0][1])**2 + (line[1][0] - line[0][0])**2)"""

def getMousePos(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)

if __name__ == "__main__":
    aWeight = 0.5

    camera = cv.VideoCapture(0)
    time.sleep(1)
    top, right, bottom, left = 10, 350, 400, 650
    num_fingers = 0
    num_frames = 0

    while(True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width = 700)
        frame = cv.flip(frame, 1)

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        if num_frames < 10:
            run_avg(gray, aWeight)
            cv.circle(frame, (int(height / 2), int(width / 2)), 30, (0, 0, 255))
        else:
            cv.imshow("background", bg/255)
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                #if cv.countNonZero(thresholded) > ((top - bottom) * (left - right) * 0.95):
                #    time.sleep(0.5)
                #    bg = None
                #    num_frames = 0
                cv.drawContours(frame, [segmented + (right, top)], -1, (0, 0, 255))
                convex_hull = cv.convexHull(segmented + (right, top), returnPoints = False)
                hull = rough_hull(convex_hull, segmented, 40)
                
                #remove bottom two points
                #del hull[hull[0][:, :, 1].argmin()[0]]
                #del hull[hull[0][:, :, 1].argmin()[1]]

                print(hull)
                hull_sorted = sorted(hull, key = lambda a : a[1],reverse = True)

                start_points = [
                    (365, 235), #thumb
                    (425, 125), #index
                    (500, 105), #middle
                    (560, 130), #ring
                    (625, 210) #pinky
                ]

                activated = []

                for point in range(5):
                    activated.append(False)
                    for pt in hull_sorted:
                        if math.hypot(start_points[point][0] - segmented[pt][0][0][0] - right, start_points[point][1] - segmented[pt][0][0][1] - top) < 25:
                            activated[point] = True
                    if activated[point]:
                        cv.circle(frame, start_points[point], 30, (255, 0, 0), thickness = -1)
                    else:
                        cv.circle(frame, start_points[point], 30, (255, 0, 0))

                num_fingers = 0

                for active in activated:
                    if active:
                        num_fingers += 1

                for point in hull_sorted:
                    cv.circle(frame, (segmented[point][0][0][0] + right, segmented[point][0][0][1] + top), 25, (0, 0, 255))

                """defects = defect_vertices(segmented, convex_hull)

                if len(defects) > 0:
                    for vert in defects:
                        cv.circle(frame, (vert[2][0][0] + right, vert[2][0][1] + top), 5, (0, 0, 255)) #points are stored in layers of nested arrays for some reason
                        cv.circle(frame, (vert[1][0][0] + right, vert[1][0][1] + top), 5, (0, 0, 255))
                        cv.circle(frame, (vert[0][0][0] + right, vert[0][0][1] + top), 5, (255, 0, 0))"""

                cv.drawContours(frame, [cv.convexHull(segmented + (right, top), segmented, 5)], -1, (0, 0, 255))
                cv.imshow("Thresholded", thresholded)

        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1

        cv.setMouseCallback("Video Feed", getMousePos)
        cv.imshow("Video Feed", frame)

        keypress = cv.waitKey(10) & 0xFF

        if keypress == ord("q"):
            break
        if keypress == ord("r"):
            num_frames = 0
            bg = None
            time.sleep(0.1)
        if keypress == ord("s"):
            enders_keyboard.search(num_fingers)

camera.release()
cv.destroyAllWindows()