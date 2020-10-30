import numpy as np
import cv2
import sys
import time
import datetime as dt

# insert time check
now = time.localtime() # 기기 시간
nowTime = [now.tm_hour, now.tm_min, now.tm_sec]
print("현재 시간 - ",nowTime[0],":",nowTime[1],":",nowTime[2])

# video
cap = cv2.VideoCapture('./Contours_Before.avi')
time.sleep(2)

# save
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./Contours_After.avi', fourcc, 30.0, (width,height))

# resize
def resizeDown(img):
    downImg = cv2.resize(img, (int(width/2), int(width/2)), interpolation=cv2.INTER_AREA)
    return downImg 
def resizeUp(img):
    upImg = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return upImg

# 윤곽 감지
fgbg = cv2.createBackgroundSubtractorMOG2()
def detect(frame):
    fgmask = fgbg.apply(frame)
    (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 600:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        roi_frame = frame[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(roi_frame,(31, 31), 0)
        frame[y:y+h, x:x+w] = blur
    return frame

# loop
while (cap.isOpened):
    ret, frame = cap.read()
    if ret == True:
        # resize 수정
        frame = resizeDown(frame)
        blurVideo = detect(frame)
        # resize 수정
        blurVideo = resizeUp(blurVideo)
        out.write(blurVideo)
        # cv2.imshow('blurVideo', blurVideo)
    else:
        break
    # VideoMove
    # try:
    #     if videoMove == True:
    #         print("Move !")
    #         videoMove = False
    #         cv2.moveWindow('blurVideo', 100, 100)
    # except Exception as e:
    #     pass
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# time check
now = time.localtime() # 기기 시간
nowTime2 = [now.tm_hour, now.tm_min, now.tm_sec]
print("현재 시간 - ",nowTime2[0],":",nowTime2[1],":",nowTime2[2])

out.release()
cap.release()
cv2.destroyAllWindows()