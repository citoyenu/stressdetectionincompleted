import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
import math
import mediapipe as mp

cap = cv2.VideoCapture('Test.mp4')

folderpath = "nervous hands"
mylist = os.listdir(folderpath)
overlaylist = []

for impath in mylist:
    image = cv2.imread(f'{folderpath}/{impath}')
    overlaylist.append(image)


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
lmlist = []
tipIds = [4,8,12,16,20]
while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.waitKey(1)
    h, w, c = overlaylist[2].shape
    #frame[0:h, 0:w] = overlaylist[2]
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id,cx,cy])
                if id == 0:
                    cv2.circle(frame, (cx,cy), 15, (255,0,0), cv2.FILLED)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    if len(lmlist) != 0:
        fingers = []
        for id in range(0,5):
            if lmlist [tipIds[id]][2] > lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        totalfingers = fingers.count(1)
        print(totalfingers)
    if ret == True:
        cv2.imshow('UwU',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print(lmlist)
cap.release()
cv2.destroyAllWindows()