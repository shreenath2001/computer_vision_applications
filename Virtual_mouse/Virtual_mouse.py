import mediapipe as mp
import cv2
import numpy as np
import time
import sys
sys.path.append('../')
from Hand_Tracking import handTracking as htm
import pyautogui 

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
wScreen, hScreen = pyautogui.size()
#print(wScreen, hScreen)
frameR = 100
smoothening = 7

pTime = 0
cTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.detectHands(img)
    lmList = detector.findPosition(img)
    if(len(lmList)!=0):
        bbox = detector.bbox(img)
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (0 ,255, 0), 2, 2)
        if fingers[1] == 1 and fingers[2] == 0:  # moving mode
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScreen))

            clocX = plocX + (x3 -  plocX) / smoothening
            clocY = plocY + (y3 -  plocY) / smoothening

            pyautogui.moveTo(clocX, clocY)
            cv2.circle(img, (x1, y1), 15,  (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:  # clicking mode
            length, img, lineInfo = detector.findDistance(8, 12, img)
            #print(length)
            if(length<40):
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15,  (0, 255, 0), cv2.FILLED)
                pyautogui.click()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()