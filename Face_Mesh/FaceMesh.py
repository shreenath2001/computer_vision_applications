import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, mode = False, maxFaces = 2, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.maxFaces = maxFaces
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFaces, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def detectFaceMesh(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        faces = []

        if self.results.multi_face_landmarks:

            for face_lms in self.results.multi_face_landmarks:

                self.mpDraw.draw_landmarks(img, face_lms, self.mpFaceMesh.FACE_CONNECTIONS,
                landmark_drawing_spec = self.drawSpec, connection_drawing_spec = self.drawSpec)

                face = []
                for id, lms in enumerate(face_lms.landmark):
                    h,w,c = img.shape
                    x,y = int(lms.x*w), int(lms.y*h)
                    face.append([id, x, y])
                faces.append(face)

        return img, faces


def main():

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        
        img, faces = detector.detectFaceMesh(img)

        if(len(faces)!=0):
            print(faces[0])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Face Mesh", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()