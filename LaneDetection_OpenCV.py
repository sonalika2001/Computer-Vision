import cv2 as cv
import numpy as np

def canny_edge(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) #hsv


    lower_white = (0, 0, 198)
    upper_white = (145, 60, 255)

    mask_white = cv.inRange(hsv, lower_white, upper_white) #masking
    result_white = cv.bitwise_and(frame, frame, mask=mask_white)
    canny = cv.Canny(result_white,10,150)   #cannyedgedetection

    return canny

def roi_pll(frame):
    height = frame.shape[0]
    width = frame.shape[1]

    pll = np.array([
                     [(0,0.5*height),(0.42*width,0.25*height),(0.55*width,0.25*height),(width,0.5*height),(width,height),(0,height)]
                   ],dtype=np.int32)
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, pll, 255)
    roi = cv.bitwise_and(frame,mask)
    return roi

def draw_lines(frame,img):
    lines = cv.HoughLinesP(img,1,np.pi/180,40,minLineLength=100,maxLineGap=70)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    return frame

cap = cv.VideoCapture("lane_vgt.mp4")
while(cap.isOpened()):
     ret, frame = cap.read()
     canny = canny_edge(frame)
     cv.imshow("canny", canny)
     roi = roi_pll(canny)
     cv.imshow("roi", roi)
     output = draw_lines(frame,roi)
     cv.imshow("output",output)
     if cv.waitKey(10) & 0xFF == ord('q'):
          break
