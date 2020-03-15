import cv2
import numpy as np
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import time

#img = cv2.imread(r'C:\Users\dell\PycharmProjects\untitled\venv\caine.jpg')

cap = WebcamVideoStream(r'D:\boschmobchl\AndreiBosch\Advanced-Lane-Detection\output1.mp4').start()
# cap = cv2.VideoCapture(src=0)
time.sleep(1)
fps = FPS().start()

while True:
    img = cap.read()
    width = int(img.shape[1] * 50 / 100)
    height = int(img.shape[0] * 50 / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    frame = resized_img
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)

    lower_white = np.array([[180, 180, 180]])
    upper_white = np.array([[255, 255, 255]])

    maskwhite = cv2.inRange(blurred, lower_white, upper_white)
    reswhite = cv2.bitwise_and(blurred, blurred, mask = maskwhite)

    cv2.imshow('rezultat', reswhite)
    k = cv2.waitKey(7) & 0xff
    if k == 27:
        break
    fps.update()

fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.stop()
cv2.destroyAllWindows()

#vid = D:\boschmobchl\AndreiBosch\Advanced-Lane-Detection\output1.mp4

#img_hsv= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#img_hsv_bgr= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#img_Gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#sobelx = cv2.Sobel(img_Gray,cv2.CV_64F,1,0,ksize=3)
#sobely = cv2.Sobel(img_Gray,cv2.CV_64F,0,1,ksize=3)

#laplacian = cv2.Laplacian(img_Gray,cv2.CV_64F)

#cv2.imwrite('savedcaine.jpg', img)

#cv2.imshow('caine', img)
#cv2.waitKey(0)

#blurred = cv2.GaussianBlur(img, (5, 5), 0)
#cv2.imshow('blurred ', blurred)
#cv2.waitKey(0)
#blurred_duplicat = blurred


#lower_red = np.array([[130, 10, 145]])
#upper_red = np.array([[255, 130, 255]])


#maskred = cv2.inRange(blurred, lower_red, upper_red)
#resred = cv2.bitwise_and(blurred, blurred, mask = maskred)
#if np.any(resred>60):
#    print('THIS PART IS RED')

#kernel = np.ones((6, 6), np.uint8)


#cnts = cv2.findContours(blurred_duplicat.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]



#mask = cv2.erode(blurred_duplicat, kernel, cv2.BORDER_REFLECT, iterations = 20)
#mask = cv2.dilate(mask, None, iterations=2)

#cv2.rectangle(blurred_duplicat, (0, 0), (0+100, 0+100), (0, 0, 0), 1)
#line = np.array([[100,110],[200,110]], np.int32).reshape((-1,1,2))
#cv2.polylines(blurred_duplicat, [line], False, (1, 1, 1), thickness=5)

#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(blurred_duplicat, 'Alba Iulia', (0+20, 0+20), font, 1.00, (1 ,1 ,0), 3)

#blurred2 = cv2.GaussianBlur(resred, (11, 11), 0)
#blurred3 = cv2.GaussianBlur(resred, (5, 5), 0)

#cv2.imshow('resred _ ', resred)
#cv2.waitKey(0)

#cv2.imshow('dblurred_duok ', blurred_duplicat)
#cv2.waitKey(0)

#cv2.imshow('resblurr ', blurred)
#cv2.waitKey(0)

#cv2.imshow('resblurr2 ', blurred2)
#cv2.waitKey(0)

#cv2.imshow('resblurrdupa', blurred3)
#cv2.waitKey(0)

#v2.imshow('caine1', img_hsv)
#cv2.waitKey(0)
#cv2.imshow('cine2', img_hsv_bgr)
#cv2.waitKey(0)
#cv2.imshow('caine3', img_Gray)
#cv2.waitKey(0)
#cv2.imshow('caine4', sobelx)
#cv2.waitKey(0)
#cv2.imshow('caine5', sobely)
#cv2.waitKey(0)
#cv2.imshow('caine6', laplacian)
#cv2.waitKey(0)

#cv2.imwrite()