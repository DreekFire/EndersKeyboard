import cv2 as cv
import numpy as np
import imutils
import time

bg = None

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold = 12):
    kernel = np.ones((5,5),np.uint8)

    global bg
    diff = cv.absdiff(bg.astype("uint8"), image)

    thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]
    thresholded = cv.dilate(thresholded, kernel, 10)
    thresholded = cv.erode(thresholded, kernel, 20)

    (_, conts, _) = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if len(conts) == 0:
        return
    else:
        segmented = max(conts, key=cv.contourArea)
        return (thresholded, segmented)

if __name__ == "__main__":
    aWeight = 0.5

    camera = cv.VideoCapture(0)
    time.sleep(3)
    top, right, bottom, left = 10, 350, 400, 650

    num_frames = 0

    while(True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width = 700)
        frame = cv.flip(frame, 1)

        clone = frame.copy()
        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            cv.imshow("background", bg/255)
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
                #if cv.countNonZero(thresholded) > ((top - bottom) * (left - right) * 0.95):
                #    time.sleep(0.5)
                #    bg = None
                #    num_frames = 0
                cv.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                hull = []
                hull.append(cv.convexHull(segmented + (right, top)))
                cv.drawContours(clone, hull, -1, (0, 0, 255))
                cv.imshow("Thresholded", thresholded)

        cv.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1

        cv.imshow("Video Feed", clone)

        keypress = cv.waitKey(10) & 0xFF

        if keypress == ord("q"):
            break
        if keypress == ord("r"):
            num_frames = 0
            bg = None
            time.sleep(1)

camera.release()
cv.destroyAllWindows()