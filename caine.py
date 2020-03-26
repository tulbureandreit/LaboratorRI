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

sx_thresh = (100, 255)
while True:
    img = cap.read()
    width = int(img.shape[1] * 50 / 100)
    height = int(img.shape[0] * 50 / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


    frame = resized_img
    frame = frame[int(height/2):height, 0:0+width]


    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    #cv2.imshow('verif_img_', frame)
    #cv2.waitKey(0)
    #hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS).astype(np.float)
    #cv2.imshow('verif_img_hls', hls)
    #cv2.waitKey(0)

    #h_channel = hls[:, :, 0]
    #l_channel = hls[:, :, 1]
    #s_channel = hls[:, :, 2]

    lower_white = np.array([[144, 144, 144]])
    upper_white = np.array([[255, 255, 255]])

    maskwhite = cv2.inRange(blurred, lower_white, upper_white)
    reswhite = cv2.bitwise_and(blurred, blurred, mask = maskwhite)

    sobelx = cv2.Sobel(reswhite, cv2.CV_64F, 1, 0, ksize=1)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1
    sxbinarye = sxbinary * 255  # (0, 255) cv2.imshow afiseaza doar in intervalul 0,255
    #daca ii dai intervalul 0,1, ea o sa iti dea inapoi imaginea in negru complet

    combined_binary = np.zeros_like(sxbinary)

    combined_binary[((reswhite >= 1) & (sxbinary == 1))] = 1
    # combined_binary[((g_binary == 1) & (sxbinary == 1))] = 1
    totalbinerye = combined_binary * 255


    cv2.imshow('verif_blurr', blurred)
    cv2.imshow('verif_final', totalbinerye)
    #cv2.imshow('verif_imagine',sxbinarye)
    #cv2.imshow('rezultat', reswhite)
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


'''

binary_img = pipeline(img)

    undist = cv2.undistort(binary_img, mtx, dist, None, mtx)
    #cv2.imshow('verif_img_binara_undst',undist)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    img_size = (undist.shape[1], undist.shape[0])

    src = np.float32([[img_size[0]-20, img_size[1]/2+50], [img_size[0], img_size[1]-30], [0, img_size[1]-30], [20, img_size[1]/2+50]])

    offset = 10
    dst = np.float32([[img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]],
                      [offset, img_size[1]], [offset, 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    # atentia la IMG SIZE, paote se strica de aici. M transform matrix
    top_down = cv2.warpPerspective(undist, M, img_size)
    top_downe = top_down*255
    cv2.imshow('verif_top_down_birdseye.jpg', top_downe)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    # incarcam o imagine chessboard pt calibrare. o incarcam intr-o lista ca sa o folosim mai tarziu
cal_image_loc = glob.glob('camera_cal/*.jpg')
calibration_images = []

for fname in cal_image_loc:
    img = mpimg.imread(fname)
    calibration_images.append(img)

# in object points sunt coordonatele colturilor reale ale tablei de sah
objp = np.zeros((12 * 12, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# punctele tablei de sah se vor salva aici
objpoints = []
imgpoints = []  # in img points se salveaza colturile tablei de sah imaginare


# gasim punctele
for image in calibration_images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (12, 12), None)  # 9 si 6 sunt pattern sizes
    if ret is True:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(image, (12, 12), corners, ret)

#  traiasca opencv
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# gray.shape e img size si none is flagurile
# print(gray.shape)

'''