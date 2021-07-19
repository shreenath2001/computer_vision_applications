import mediapipe as mp
import cv2
import numpy as np
import time
import sys
sys.path.append('../')
from Hand_Tracking import handTracking as htm
import os

imgpath = 'static'
imgList = os.listdir(imgpath)
#print(imgList)
overlayList = []
for path in imgList:
    img = cv2.imread(f'{imgpath}/{path}')
    overlayList.append(img)
#print(len(overlayList))
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

pTime = 0
cTime = 0
drawColor = (255, 0, 255)
brushThickness = 15
eraserThickness = 100
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
imgInv = np.zeros((720, 1280, 3), np.uint8)

detector = htm.HandDetector(maxHands=1, detectionCon=0.75)

while(True):
    success, img = cap.read()
    #img = cv2.resize(img, (1280, 720))
    img = cv2.flip(img, 1)

    img = detector.detectHands(img)
    lmList = detector.findPosition(img)
    if(len(lmList)!=0):

        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        #print(fingers)

        if fingers[1] and fingers[2]:   #selection
            xp, yp = 0, 0
            if y1<125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)
            #print("Selection")


        if fingers[1] and fingers[2] == False:   # drawing mode
            cv2.circle(img, (x1, y1), 15,  drawColor, cv2.FILLED)
            #print("Drawing")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    img[0:header.shape[0], 0:header.shape[1]] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Virtual Painter", img)
    #cv2.imshow("Virtual Painter Canvas", imgCanvas)
    #cv2.imshow("Virtual Painter Inv", imgInv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()