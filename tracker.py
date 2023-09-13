import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo_50epoch.pt")



def RGB(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x,y]
        
        print(colorsBGR)

def calc_centre(x1,y1,x2,y2):
    return (x1+x2)/2 , (y1+y2)/2
    

def isValid(centre_x,centre_y):
    y_max = 0.248 * centre_x + 157
    y_min  = 0.186 * centre_x + 90.7
    if((centre_y > y_min) and (centre_y < y_max)):
        return True
    else:
        return False
def bounding_box(result,frame):
    m = result[0].boxes.data
    tensor_data = torch.tensor(m)
    numpy_array = tensor_data.cpu().detach().numpy()
    #numpy_matrix = numpy_array.reshape(2, 6)
    for i in range(numpy_array.shape[0]):
        x1 = int(numpy_array[i][0])
        y1 = int(numpy_array[i][1])
        x2 = int(numpy_array[i][2])
        y2 = int(numpy_array[i][3])
        x_cen,y_cen = calc_centre(x1,y1,x2,y2)
        if isValid(x_cen,y_cen):
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0), 2)
       

        
    


    
    

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
    result = model.predict(frame, conf = 0.01,iou = 0.03)
    
    cv2.line(frame,(0, 157),(712, 334),(255,0,0),thickness= 5)
    cv2.line(frame,(126, 114),(712, 222),(0,0,255),thickness= 5)
    bounding_box(result,frame)
    cv2.imshow('',frame)

    
    
    if cv2.waitKey(1)&0xFF == 27:
        break
