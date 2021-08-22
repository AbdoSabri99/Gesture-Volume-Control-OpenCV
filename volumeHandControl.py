import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
##########################################
wCam,hCam=640,480
vol,volBar=0,480
volPer=0
##########################################

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0
detector=htm.HandDetector(minDetCon=0.7)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRang=volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)
minVol=volRang[0]
maxVol=volRang[1]
while True:
    success,img=cap.read()
    img=detector.findHand(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        #print(lmList[4])
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(x1,y1),5,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),5,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        length=math.hypot(x1-x2,y1-y2)
        #print(length)
        #hand range 10-120
        vol=np.interp(length,[9,130],[minVol,maxVol])
        volBar=np.interp(length,[9,130],[400,150])
        volPer=np.interp(length,[9,130],[0,100])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
        cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
        cv2.putText(img,f'{int(volPer)} %',(48,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{int(fps)}',(40,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    cv2.imshow("cam",img)
    cv2.waitKey(1)
