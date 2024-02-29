import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import imutils

import numpy as np
from ultralytics import YOLO
from collections import defaultdict

color = (0,255,0)
color_red = (0,0,255)
thickness = 2

font_scale = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

video_path = "inference/test.mp4"
model_path = "models/yolov8n.pt"

cap = cv2.VideoCapture(video_path)
model = YOLO(model_path)

width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("video.avi", fourcc, 20.0, (width, height))


vehicle_ids = [2, 3, 5, 7]
track_history = defaultdict(lambda: [])

up = {}
down = {}
threshold = 450

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame = imutils.resize(frame, width=1280)
    results = model.track(frame, persist=True, verbose=False)[0]
    
    # track_ids = results.boxes.id.int().cpu().tolist()
    bboxes = np.array(results.boxes.data.tolist(), dtype="int") #xyxy
    
    cv2.line(frame, (0, threshold), (1280, threshold), color_red, thickness)
    cv2.putText(frame, "Reference Line", (620, 445), font, 0.7, color_red, thickness)
    
    
    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        
        if class_id in vehicle_ids:
            class_name = results.names[int(class_id)].upper() # car --> CAR
            
            # [INFO]... CTRL + K + C // CTRL + K + U
            # print("BBoxes: ", (x1, y1, x2, y2))
            # print("Class: ", class_name)
            # print("ID: ", track_id)
            
            track = track_history[track_id]
            track.append((cx, cy))
            if len(track) > 15:
                track.pop(0)
            
            points = np.hstack(track).astype("int32").reshape((-1,1,2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            text = "ID:{} {}".format(track_id, class_name)
            cv2.putText(frame, text, (x1, y1-5), font, font_scale, color, thickness)
            
            if cy>threshold-5 and cy<threshold+5 and cx<670:
                down[track_id] = x1, y1, x2, y2
                
            if cy>threshold-5 and cy<threshold+5 and cx>670:
                up[track_id] = x1, y1, x2, y2
                
        
        print("UP Dictionary Keys: ", list(up.keys()))
        print("DOWN Dictionary Keys: ", list(down.keys()))
        
        up_text = "Giden:{}".format(len(list(up.keys())))
        down_text = "Gelen:{}".format(len(list(down.keys())))
        
        cv2.putText(frame, up_text, (1150, threshold-5), font, 0.8, color_red, thickness)
        cv2.putText(frame, down_text, (0, threshold-5), font, 0.8, color_red, thickness)
    
    writer.write(frame)
    cv2.imshow("Test", frame)
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("[INFO].. The video was successfully processed/saved !")