import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm

brushThickness = 25
eraserThickness = 100

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    if imPath.endswith(('.jpg','.png')):
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
model_path = "hand_landmarker.task"
detector = htm.handDetector(model_path=model_path, maxHands=1)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

timestamp_ms = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    timestamp_ms += 1

    img = detector.findHands(img, timestamp_ms=timestamp_ms)
    lmList = detector.lmList

    if lmList:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
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
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = brushThickness
            if drawColor == (0, 0, 0):
                thickness = eraserThickness

            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)

            xp, yp = x1, y1

        if all(fingers):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()