import mediapipe as md
import cv2
import numpy as np
import time
import sys
sys.path.append('../')
from Pose_Tracking import PoseTracking as ptm

#window_name = "window"
cap = cv2.VideoCapture(0)
#cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

pTime = 0
cTime = 0
count = 0
dir = 0

detector = ptm.PoseDetector()

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    #img = cv2.imread("AiTrainer/test.jpg")
    img = detector.detectPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        # right arm
        #angle = detector.findAngle(img, 12, 14, 16)
        # left arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (210, 310), (650, 100))
        #print(angle, per)
        color = (255, 0, 0)
        if per == 100:
            if dir == 0:
                count+=0.5
                dir = 1
                color = (0, 255, 0)
        if per == 0:
            if dir == 1:
                count+=0.5
                dir = 0
                color = (0, 0, 255)
        #print(count)
        # Bar rectangle
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1050, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
        # show counter
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("AI Trainer", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()