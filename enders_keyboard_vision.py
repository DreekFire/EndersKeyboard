import cv2 as cv
import numpy as np
import imutils
import time
import math
import enders_keyboard
import enders_keyboard_output

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

def get_mouse_pos(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)

def vector_proj(v1, v2):
    return np.multiply((np.dot(v1, v2) / np.dot(v1, v1)), v1)

def simulate_key(key):
    enders_keyboard_output.type_key(key)

if __name__ == "__main__":
    aWeight = 0.5

    key_dict = {
        0 : "a",
        1 : "b",
        2 : "c",
        3 : "d",
        4 : "e",
        5 : "f",
        6 : "g",
        7 : "h",
        8 : "i",
        9 : "j",
        10 : "k",
        11 : "l",
        12 : "m",
        13 : "n",
        14 : "o",
        15 : "p",
        16 : "q",
        17 : "r",
        18 : "s",
        19 : "t",
        20 : "u",
        21 : "v",
        22 : "w",
        23 : "x",
        24 : "y",
        25 : "z"
    }

    camera = cv.VideoCapture(0)
    time.sleep(1)
    top, right, bottom, left = 10, 350, 350, 750
    num_fingers = 0
    num_frames = 0

    start_points = [
        (385, 235), #thumb
        (425, 125), #index
        (500, 105), #middle
        (560, 130), #ring
        (615, 210) #pinky
    ]

    start_center = (0, 0)

    current_points = start_points.copy()

    act = False
    last_found = [True, True, True, True, True]

    while(True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width = 700)
        frame = cv.flip(frame, 1)

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        inner = [False, False, False, False, False]
        outer = [False, False, False, False, False]

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
                
                #cv.drawContours(frame, [segmented + (right, top)], -1, (0, 0, 255))
                convex_hull = cv.convexHull(segmented + (right, top), returnPoints = False)
                hull = rough_hull(convex_hull, segmented, 40)
                
                #remove bottom two points
                #del hull[hull[0][:, :, 1].argmin()[0]]
                #del hull[hull[0][:, :, 1].argmin()[1]]

                if len(segmented) > 0:
                    hull_sorted = sorted(hull, key = lambda a : segmented[a[0]][0][1])
                    hull_sorted = hull_sorted[:min(len(hull_sorted), 5)]

                activated = []
                if act is False:
                    for point in range(5):
                        activated.append(False)
                        for pt in hull_sorted:
                            if math.hypot(start_points[point][0] - segmented[pt][0][0][0] - right, start_points[point][1] - segmented[pt][0][0][1]) < 25:
                                activated[point] = True
                        cv.circle(frame, (start_points[point][0], start_points[point][1]), 30, (255, 0, 0), thickness = -1 if activated[point] else 1)

                    num_fingers = 0

                    for active in activated:
                        if active is True:
                            num_fingers += 1
                    if num_fingers >= 5:
                        print("act")
                        act = True
                        m = cv.moments(segmented)
                        start_center = (int(m["m10"] / m["m00"]) + right, int(m["m01"] / m["m00"]) + top)
                else:
                    m = cv.moments(segmented)
                    current_center = (int(m["m10"] / m["m00"]) + right, int(m["m01"] / m["m00"]) + top)
                    center_diff = np.subtract(current_center, start_center)
                    for point in range(len(start_points)):
                        start_points[point] = np.add(start_points[point], center_diff)
                        start_center = current_center

                    thumb_inner = False
                    thumb_outer = False
                    found = [False, False, False, False, False]
                    for point in range(5):
                        vect = [start_points[point][0] - current_center[0], start_points[point][1] - current_center[1]]
                        mag = math.sqrt(vect[0]**2 + vect[1]**2)
                        inner[point] = False
                        outer[point] = False
                        for pt in hull_sorted:
                            if math.hypot(current_points[point][0] - segmented[pt][0][0][0] - right, current_points[point][1] - segmented[pt][0][0][1]) < 20 and math.hypot(start_points[point][0] - segmented[pt][0][0][0] - right, start_points[point][1] - segmented[pt][0][0][1]) < 40:
                                current_points[point] = (segmented[pt][0][0][0] + right, segmented[pt][0][0][1])
                                diff = np.subtract((current_points[point][0], current_points[point][1]), start_points[point])
                                adjusted_pt = np.add(start_points[point], vector_proj(vect, diff))
                                cv.circle(frame, (int(adjusted_pt[0]), int(adjusted_pt[1])), 30, (255, 0, 0), thickness = -1)
                                found[point] = True
                        if (not found[point]) and found[point] is not last_found[point]:
                            d = math.hypot(current_points[point][0] - current_center[0], current_points[point][1] - current_center[1])
                            current_points[point] = start_points[point]
                            if d < mag:
                                inner[point] = True
                                outer[point] = False
                            else:
                                inner[point] = False
                                outer[point] = True
                            cv.circle(frame, (current_points[point][0], current_points[point][1]), 25, (255, 0, 255))
                        last_found[point] = found[point]
                        cv.circle(frame, (start_points[point][0], start_points[point][1]), 30, (0, 255, 0), thickness = 1)
                        cv.line(frame, (start_points[point][0], start_points[point][1]), (int(start_points[point][0] + vect[0] * 15 / mag), int(start_points[point][1] + vect[1] * 15 / mag)), (0, 255, 0), thickness = 1)
                    cv.circle(frame, current_center, 25, (0, 0, 255))
                    thumb_dist = math.hypot(current_points[0][0] - current_center[0], current_points[0][1] - current_center[1])
                    thumb_vect = [start_points[0][0] - start_center[0], start_points[0][1] - start_center[1]]
                    thumb_mag = math.sqrt(vect[0]**2 + vect[1]**2)
                    if thumb_dist - thumb_mag < -15:
                        thumb_inner = True
                        thumb_outer = False
                    elif thumb_dist - thumb_mag > 15:
                        thumb_outer = True
                        thumb_inner = False
                    
                    for finger in range(len(inner)):
                        if inner[finger] is True:
                            simulate_key(key_dict[(8 if thumb_inner else (16 if thumb_outer else 0)) + finger * 2])
                        elif outer[finger] is True:
                            simulate_key(key_dict[(8 if thumb_inner else (16 if thumb_outer else 0)) + finger * 2 + 1])

                """defects = defect_vertices(segmented, convex_hull)

                if len(defects) > 0:
                    for vert in defects:
                        cv.circle(frame, (vert[2][0][0] + right, vert[2][0][1] + top), 5, (0, 0, 255)) #points are stored in layers of nested arrays for some reason
                        cv.circle(frame, (vert[1][0][0] + right, vert[1][0][1] + top), 5, (0, 0, 255))
                        cv.circle(frame, (vert[0][0][0] + right, vert[0][0][1] + top), 5, (255, 0, 0))"""

                #cv.drawContours(frame, [cv.convexHull(segmented + (right, top), segmented, 5)], -1, (0, 0, 255))
                cv.imshow("Thresholded", thresholded)

        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1

        cv.setMouseCallback("Video Feed", get_mouse_pos)
        cv.imshow("Video Feed", frame)

        keypress = cv.waitKey(10) & 0xFF

        if keypress == ord("q"):
            break
        if keypress == ord("r"):
            num_frames = 0
            start_points = [
                (385, 235), #thumb
                (425, 125), #index
                (500, 105), #middle
                (560, 130), #ring
                (615, 210) #pinky
            ]
            act = False
            bg = None
            time.sleep(0.1)
        if keypress == ord("s"):
            enders_keyboard.search(num_fingers)

camera.release()
cv.destroyAllWindows()