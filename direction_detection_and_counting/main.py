import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import imutils

import numpy as np
from ultralytics import YOLO
from collections import defaultdict


blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
black = (0,0,0)
white = (255,255,255)

thickness = 2
font_scale = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

up = {}
down = {}
left = {}
right = {}

vehicles = [2, 3, 5, 7]

polygon_up = np.array([[467, 295],[578, 370],
                       [710, 361],[559, 283]], np.int32)

polygon_down = np.array([[760, 610],[490, 660],
                         [527, 714],[890, 711]], np.int32)

polygon_left = np.array([[107, 451],[213, 438],
                         [220, 469],[85, 481]], np.int32) # ÖDEV: Alanı düzenle.

polygon_right = np.array([[990, 453],[1262, 404],
                          [1276, 467],[1072, 506]], np.int32)


video_path = "inference/intersection.mp4"

cap = cv2.VideoCapture(video_path)
model = YOLO("models/yolov8n.pt")

track_history = defaultdict(lambda: [])

width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("video.avi", fourcc, 20.0, (width, height))


while True:
    ret, frame = cap.read()
    if ret==False:
        break
    
    frame = imutils.resize(frame, width=1280)
    
    cv2.polylines(frame, [polygon_up], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame, [polygon_down], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame, [polygon_left], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame, [polygon_right], isClosed=True, color=green, thickness=thickness)
    
    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")
    
    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        
        if class_id in  vehicles:
            cv2.circle(frame, (cx, cy), 3, blue, -1)
            cv2.rectangle(frame, (x1,y1), (x2,y2), blue, thickness=1) 
            
            class_name = results.names[class_id].upper()
            text = "ID:{} {}".format(track_id, class_name)
            cv2.putText(frame, text, (x1, y1-5), font, font_scale, blue, thickness)
            
            track = track_history[track_id]
            track.append((cx,cy))
            
            if len(track)>15:
                track.pop(0)
                
            points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
            cv2.polylines(frame, [points], isClosed=False, color=blue, thickness=thickness)
            
            up_result = cv2.pointPolygonTest(polygon_up, (cx, cy), measureDist=False)
            down_result = cv2.pointPolygonTest(polygon_down, (cx, cy), measureDist=False)
            left_result = cv2.pointPolygonTest(polygon_left, (cx, cy), measureDist=False)
            right_result = cv2.pointPolygonTest(polygon_right, (cx, cy), measureDist=False)
            
            if up_result > 0:
                # print("UP!")
                up[track_id] = x1, y1, x2, y2
            
            if down_result > 0:
                # print("DOWN!")
                down[track_id] = x1, y1, x2, y2
                
            if left_result > 0:
                # print("LEFT!")
                left[track_id] = x1, y1, x2, y2
                
            if right_result > 0:
                # print("RIGHT!")
                right[track_id] = x1, y1, x2, y2
            
            
            
            
    
    # print("Up Direction Counter: ", len(list(up.keys())))
    # print("Down Direction Counter: ", len(list(down.keys())))
    # print("Left Direction Counter: ", len(list(left.keys())))
    # print("Right Direction Counter: ", len(list(right.keys())))
    
    up_counter_text = "Up Direction Counter: {}".format(str(len(list(up.keys()))))
    down_counter_text = "Down Direction Counter: {}".format(str(len(list(down.keys()))))
    left_counter_text = "Left Direction Counter: {}".format(str(len(list(left.keys()))))
    right_counter_text = "Right Direction Counter: {}".format(str(len(list(right.keys()))))
    
    cv2.rectangle(frame, (0,0), (350, 150), white, -1) 
    cv2.putText(frame, up_counter_text, (10, 25), font, 0.8, black, thickness)
    cv2.putText(frame, down_counter_text, (10, 65), font, 0.8, black, thickness)
    cv2.putText(frame, left_counter_text, (10, 105), font, 0.8, black, thickness)
    cv2.putText(frame, right_counter_text, (10, 145), font, 0.8, black, thickness)
    
    writer.write(frame)
    cv2.imshow("Direction Counter", frame)
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break
    
cap.release()
writer.release()

cv2.destroyAllWindows()
print("[INFO]...Processing is finished!")
