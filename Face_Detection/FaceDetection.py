import mediapipe as mp 
import cv2
import time

class FaceDetection():

    def __init__(self):

        self.mpfaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpfaceDetection.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

    def detectingFace(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)

        if results.detections:
                for id,detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    h,w,c = img.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)

        return img

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)

    detector = FaceDetection()

    while(True):

        success, img = cap.read()

        detector.detectingFace(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()