import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions import hands

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipPos = [4, 8, 12, 16, 20]

    def detectHands(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:

            for hand_lms in self.results.multi_hand_landmarks:

                self.mpDraw.draw_landmarks(img, hand_lms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo = 0, draw = False):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []
        rigthHand = (self.lmList[5][1]>self.lmList[17][1])
        if(rigthHand):
            if self.lmList[self.tipPos[0]][1] > self.lmList[self.tipPos[0]-1][1]: # right thumb
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if self.lmList[self.tipPos[0]][1] < self.lmList[self.tipPos[0]-1][1]: # left thumb
                fingers.append(1)
            else:
                fingers.append(0)
        for i in range(1, 5):
            if self.lmList[self.tipPos[i]][2] < self.lmList[self.tipPos[i]-2][2]: # 4 fingers
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, img = cap.read()
        
        detector.detectHands(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()