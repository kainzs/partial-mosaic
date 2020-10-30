from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import socket
import os
import platform

# ip 찾기
# Darwin
if platform.system() == 'Darwin':
	ip = socket.gethostbyname(socket.gethostname())
	print("현재 아이피 주소 :", ip)
	print("실행 포트번호 : 8282")

# Windows
if platform.system() == 'Windows':
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.connect(("8.8.8.8", 80))
	ip = s.getsockname()[0]
	print("현재 아이피 주소 :", ip)
	print("실행 포트번호 : 8282")

faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
bodyCascade = cv2.CascadeClassifier('./haarcascade_fullbody.xml')

outputFrame = None
lock = threading.Lock()

# 서버
app = Flask(__name__)

# 실시간 비디오
vs = VideoStream(src=0).start()
time.sleep(1.0)

# # 화면 사이즈
# width = 426
# height = 240
# # XVID 코덱으로 지정 Four Character Code
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # OUT 설정 파일명, 코덱, 프레임 수, 사이즈
# out = cv2.VideoWriter('./saveVideo.avi', fourcc, 30.0, (width,height))

# 카운트, 탐색 카운트, 비교, 탐색 데이터
dataCnt = 75
test = np.array([])
data = np.array([])

@app.route("/")
def index():
	return render_template("index.html")

def detect_motion(frameCount):
	global vs, outputFrame, lock, test, data, dataCnt

	# md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	while True:
		frame = vs.read()
		frame = imutils.resize(frame, width=426)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)
		# 날짜
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# haarcascade 탐색
		faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 10, minSize=(50, 50), flags = cv2.CASCADE_SCALE_IMAGE)
		# bodies = bodyCascade.detectMultiScale(frame)

		# 탐색 미스 방지
		try:
			if type(faces)==type(test):
				data = faces
				dataCnt = 75
			else:
				dataCnt -= 1
			if dataCnt==0:
				data = np.array([])
		except Exception as e:
			print(str(e))

		# 얼굴 블러
		# if
		# for (x, y, w, h) in data:
		# 	roiFrame = frame[y-20:y+h+20, x-20:x+w+20]
		# 	blur = cv2.GaussianBlur(roiFrame,(101, 101), 0)
		# 	frame[y-20:y+h+20, x-20:x+w+20] = blur

		## !! 반전
		for (x, y, w, h) in data:
			roiFrame = frame[y-20:y+h+20, x-20:x+w+20]
			frame = cv2.GaussianBlur(frame,(21, 21), 0)
			frame[y-20:y+h+20, x-20:x+w+20] = roiFrame
		## !! 반전

		# 바디 블러
		# for (x, y, w, h) in bodies:
		# 	roiFrame = frame[y:y+h, x:x+w]
		# 	blur = cv2.GaussianBlur(roiFrame,(101, 101), 0)
		# 	frame[y:y+h, x:x+w] = blur

		# 모션 탐색
		# if total > frameCount:
		# 	motion = md.detect(gray)
		# 	if motion is not None:
		# 		(thresh, (minX, minY, maxX, maxY)) = motion

		# 		# 모션 블러
		# 		roi_frame = frame[minY:maxY, minX:maxX]
		# 		blur = cv2.GaussianBlur(roi_frame,(101, 101), 0)
		# 		# cv2.rectangle(frame, (minX, minY), (maxX, maxY),
		# 		# 	(0, 0, 255), 2)
		# 		frame[minY:maxY, minX:maxX] = blur
		
		# md.update(gray)
		# total += 1
		with lock:
			outputFrame = frame.copy()
		
def generate():
	global outputFrame, lock
	while True:
		with lock:
			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			if not flag:
				continue
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	app.run(host=ip, port='8282', debug=True,
		threaded=True, use_reloader=False)

vs.stop()
# out.release()
