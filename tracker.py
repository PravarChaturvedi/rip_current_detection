import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo_50epoch.pt")



def RGB(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x,y]
        
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB',RGB)

cap = cv2.VideoCapture('VID20220830111033.mp4')

count  = 0
while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (720, int(720*(9/16))))
    

    
    


   #
   # 
   # 
    
    
    result = model.predict(frame, conf = 0.01,iou = 0.1)
    result_tensor = result[0].boxes
    res_plotted = result[0].plot()
    cv2.line(res_plotted,(0, 157),(712, 334),(255,0,0),thickness= 5)
    cv2.line(res_plotted,(126, 114),(712, 222),(0,0,255),thickness= 5)
    cv2.imshow("RGB",res_plotted)
    
    if cv2.waitKey(1)&0xFF == 27:
        break
