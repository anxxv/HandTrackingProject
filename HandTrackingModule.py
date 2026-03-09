import cv2
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class handDetector():
    def __init__(self, model_path, maxHands=2):
        self.maxHands = maxHands

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.maxHands
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.lmList = []

    def findHands(self, image, draw=True, timestamp_ms=0):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        self.lmList = []
        h, w, _ = image.shape

        if results.hand_landmarks:
            for hands in results.hand_landmarks:
                for id, lm in enumerate(hands):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return image

    def fingersUp(self):
        fingers = []
        if not self.lmList:
            return [0,0,0,0,0]

        if self.lmList[4][1] > self.lmList[3][1]: #большой палчик
            fingers.append(1)
        else:
            fingers.append(0)

        for id in [8,12,16,20]: #остальные
            if self.lmList[id][2] < self.lmList[id-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        if not self.lmList:
            return 0, img, None

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        length = math.hypot(x2 - x1, y2 - y1)

        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
            cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 10, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)

        return length, img, [x1,y1,x2,y2,cx,cy]

def main():
    cap = cv2.VideoCapture(0)
    model_path = "hand_landmarker.task"  # путь к модели
    detector = handDetector(model_path)

    pTime = 0
    timestamp_ms = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        timestamp_ms += 1
        img = detector.findHands(img, timestamp_ms=timestamp_ms)

        if detector.lmList: #точки
            print(detector.lmList)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()