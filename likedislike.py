from smtplib import LMTP
import mediapipe as mp
import cv2
camera = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands= mp_hands.Hands()
tipid = [8, 12, 16, 20]
thumbtip= 4
def drawhandlandmarks(img, hand_landmarks):
    if hand_landmarks:
        for hand in hand_landmarks:
            lmlist=[]
            for lm in hand.landmark:
                lmlist.append(lm)
            fingerfoldStatus=[]
            for tip in tipid:
                if lmlist[tip].x < lmlist[tip-3].x:
                    fingerfoldStatus.append(True)
                else:
                    fingerfoldStatus.append(False)
            if all (fingerfoldStatus):
                if lmlist[thumbtip].y<lmlist[thumbtip-1].y<lmlist[thumbtip-2].y:
                    cv2.putText(img ,"LIKE", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                if lmlist[thumbtip].y>lmlist[thumbtip-1].y>lmlist[thumbtip-2].y:
                    cv2.putText(img ,"DISLIKE", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec((0,0,255),2,2), 
                                        mp_drawing.DrawingSpec((0,255,0),4,2))




while True:
    success, img = camera.read()
    img = cv2.flip(img, 1)
    h, w, c= img.shape   
     
    results = hands.process(img)
    hand_landmarks = results.multi_hand_landmarks
    drawhandlandmarks(img, hand_landmarks)
    cv2.imshow("reasults", img)
    if cv2.waitKey(1) == 32:
        break
cv2.destroyAllWindows()
