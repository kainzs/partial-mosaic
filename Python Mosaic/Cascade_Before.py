import numpy as np
import cv2
import sys
import time
import datetime as dt

# time check
now = time.localtime()
nowTime = [now.tm_hour, now.tm_min, now.tm_sec]
print("현재 시간 - ",nowTime[0],":",nowTime[1],":",nowTime[2])

# video
cap = cv2.VideoCapture('./Cascade_Before.avi')
time.sleep(2)

# save
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./Cascade_After.avi', fourcc, 30.0, (width,height))

# MISS
dataCnt = 99
data = np.array([])
test = np.array([])

# detect
bodyCascade = cv2.CascadeClassifier('./haarcascade_fullbody.xml')
def detect(frame):
    global dataCnt, data, test
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bodies = bodyCascade.detectMultiScale(gray)

    try:
        if type(bodies)==type(test):
            data = bodies
            dataCnt = 99
        else:
            dataCnt -= 1
        if dataCnt == 0:
            data = np.array([])
    except Exception as e:
        print(str(e))

    # change
    try:
        for (x, y, w, h) in data:
            roi_frame = frame[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(roi_frame,(31, 31), 0)
            frame[y:y+h, x:x+w] = blur
    except Exception as e:
        print(str(e))
    return frame

# resize
def resizeDown(img):
    downImg = cv2.resize(img, (int(width/2), int(width/2)), interpolation=cv2.INTER_AREA)
    return downImg
def resizeUp(img):
    upImg = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return upImg

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
now = time.localtime()
nowTime2 = [now.tm_hour, now.tm_min, now.tm_sec]
print("현재 시간 - ",nowTime2[0],":",nowTime2[1],":",nowTime2[2])

out.release()
cap.release()
cv2.destroyAllWindows()
