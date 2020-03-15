import argparse
import time
from collections import deque
import matplotlib.pyplot as plt
import array as arr

import PID
import cv2
import imutils
import numpy as np
from imutils.video import FPS
from imutils.video import WebcamVideoStream

ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
args = vars(ap.parse_args())
height = 360
width = 480
pts = deque(maxlen=args["buffer"])

referinta_init1 = int(input('Adauga referinta: '))
referinta_init = 100-referinta_init1
referinta = referinta_init/100*360
eroare = 0
suma_erori = 0
eroare_trecuta = 0
cap = cv2.VideoCapture(0)
time.sleep(1)
fps = FPS().start()
co = []
com = arr.array('d', co)
counter = 0
kp = 2.395
ti = 160
ki = kp/ti
(dX, dY) = (0, 0)
direction = ""
comanda = ""
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, 480, 480)
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred_frame = cv2.GaussianBlur(hsv, (11, 11), 0)
    edges = cv2.Canny(frame, 100, 200)
    line = np.array([[1, referinta], [480, referinta]], np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [line], False, (255, 255, 255), thickness=1)

    colorLower = np.array([[100, 40, 100]])
    colorUpper = np.array([[180, 255, 200]])
    fgmask = fgbg.apply(frame)
    mask = cv2.inRange(blurred_frame, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    res = cv2.bitwise_and(frame, blurred_frame, mask=mask)
    res1 = cv2.bitwise_and(frame, res, mask=fgmask)
    res11 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 150:
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            pts.appendleft(center)
        for i in np.arange(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            yy = pts[i][1]
            eroare = (referinta - yy)/360
            suma_erori = suma_erori + eroare
            comandaa = kp*eroare + ki*suma_erori
            comandaa = -comandaa
            #if (comandaa >= 100):
            #    comandaa = 100
            #if comandaa < 0:
            #    comandaa = 0
            comandaa = (comandaa + abs(comandaa))/2
            comanda = str(comandaa)
            print(comanda)
            com.append(float(comandaa))
            # if yy <= referinta:
            #     comanda = "comanda = 0"
            #     comandaa = "0"
            #     com.append(float(comandaa))
            # elif yy > referinta+150:
            #     comanda = "comanda = 100"
            #     comandaa = "100"
            #     com.append(float(comandaa))
            # elif yy > referinta+135:
            #     comanda = "comanda = 90"
            #     comandaa = "90"
            #     com.append(float(comandaa))
            # elif yy > referinta+120:
            #     comanda = "comanda = 80"
            #     comandaa = "80"
            #     com.append(float(comandaa))
            # elif yy > referinta+105:
            #     comanda = "comanda = 70"
            #     comandaa = "70"
            #     com.append(float(comandaa))
            # elif yy > referinta+90:
            #     comanda = "comanda = 60"
            #     comandaa = "60"
            #     com.append(float(comandaa))
            # elif yy > referinta+75:
            #     comanda = "comanda = 50"
            #     comandaa = "50"
            #     com.append(float(comandaa))
            # elif yy > referinta+60:
            #     comanda = "comanda = 40"
            #     comandaa = "40"
            #     com.append(float(comandaa))
            # elif yy > referinta+45:
            #     comanda = "comanda = 30"
            #     comandaa = "30"
            #     com.append(float(comandaa))
            # elif yy > referinta+30:
            #     comanda = "comanda = 20"
            #     comandaa = "20"
            #     com.append(float(comandaa))
            # elif yy > referinta+15:
            #     comanda = "comanda = 10"
            #     comandaa = "10"
            #     com.append(float(comandaa))

            eroare_trecuta = eroare

    cv2.putText(frame, comanda, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 3)
    cv2.imshow('Program', frame)
    cv2.imshow('program', mask)
    k = cv2.waitKey(7) & 0xff
    if k == 27:
        break
    fps.update()


plt.plot(com)
plt.ylabel('comanda')
plt.savefig('comenzi.png')
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.stop()
cv2.destroyAllWindows()
