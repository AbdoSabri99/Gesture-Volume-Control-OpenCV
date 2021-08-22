import cv2  as cv
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,mode=False,maxHands=2,minDetCon=0.5,minTracCon=0.5):
                    self.mode=mode
                    self.maxHands=maxHands
                    self.minDetCon=minDetCon
                    self.minTracCon=minTracCon
                    self.mpHands=mp.solutions.hands
                    self.hand=self.mpHands.Hands(self.mode,self.maxHands, self.minDetCon, self.minTracCon)
                    self.mpDraw=mp.solutions.drawing_utils

    def findHand(self,img,draw=True):
            imgrgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
            self.results=self.hand.process(imgrgb)
            if self.results.multi_hand_landmarks:
                for hands in self.results.multi_hand_landmarks:
                     if draw:
                            self.mpDraw.draw_landmarks(img,hands,self.mpHands.HAND_CONNECTIONS)
            return img



    def findPosition(self,img,handNo=0,draw=True):
                lmList=[]
                if self.results.multi_hand_landmarks:
                    myHand=self.results.multi_hand_landmarks[handNo]
                    for id,lm in enumerate(myHand.landmark):
                        h,w,c=img.shape
                        cx,cy=int(lm.x*w),int(lm.y*h)
                        lmList.append([id,cx,cy])
                        if draw:
                             cv.circle(img,(cx,cy),15,(255,0,255),cv.FILLED)
                return lmList

def main():
    pTime=0
    cTime=0
    cap=cv.VideoCapture(0)
    Detector=HandDetector()
    while True:
        success,img= cap.read()
        img=Detector.findHand(img,draw=False)
        lmList=Detector.findPosition(img,draw=False)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,255),2)
        cv.imshow('image',img)
        cv.waitKey(1)

if __name__=="__main__":
    main()
