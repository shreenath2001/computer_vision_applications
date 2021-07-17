import mediapipe as md
import cv2
import time
import sys
sys.path.append('../')
from Hand_Tracking import handTracking as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

detector = htm.HandDetector(maxHands=1)
tipPos = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    img = detector.detectHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:
        fingers = []
        rigthHand = (lmList[5][1]>lmList[17][1])
        if(rigthHand):
            if lmList[tipPos[0]][1] > lmList[tipPos[0]-1][1]: # right thumb
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lmList[tipPos[0]][1] < lmList[tipPos[0]-1][1]: # left thumb
                fingers.append(1)
            else:
                fingers.append(0)
        for i in range(1, 5):
            if lmList[tipPos[i]][2] < lmList[tipPos[i]-2][2]: # 4 fingers
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        total = fingers.count(1)
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()