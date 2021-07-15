import cv2
import time
import os
import HandTrackingModule as htm

########################################
wCam, hCam = 640,480
########################################

cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)

folderPath = "Images"
myList = os.listdir(folderPath)
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    image = cv2.resize(image,(200,200))
    #print(f'{folderPath}/{imgPath}')
    overlayList.append(image)

prevTime = 0
currTime = 0
detector = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cam.read()
    img  = detector.findHands(img)
    lmList  = detector.findPosition(img, draw = False)
    #print(lmList)

    if len(lmList)!=0:
        #The idea is to get the keypoints at the tip of our fingers.
        #Based on it we determine if our finger are close or open
        #But Thumb is an exception.
        # Id 4 =>THUMB_TIP, Id 8 =>INDEX_FINGER_TIP
        # Id 12 =>MIDDLE_FINGER_TIP, Id 16 =>RING_FINGER_TIP
        # Id 20 =>PINKY_TIP
        #We check if points [4,8,12,16,20] are below [2,6,10,14,18] resp
        #For the thunb we check if it's right or left of the reference point.
        fingers = []

        #Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[4]][1]:
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]: # For left hand
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]: # For right hand
                fingers.append(1)
            else:
                fingers.append(0)

        #The remaining 4 fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]: #remebemr in cv2 image starts from the top
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        #print(totalFingers)

        h,w,c = overlayList[totalFingers-1].shape
        img[0:h,0:w] = overlayList[totalFingers-1] #(-1 position of a list takes the last element..here 6.jpg)

        cv2.rectangle(img, (28,225), (178,425), (0,255,0), cv2.FILLED)
        cv2.putText(img, f'{totalFingers}',(45,375), cv2.FONT_HERSHEY_PLAIN, 10,(255,0,0),25)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(img, f'FPS:{fps}',(400,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
