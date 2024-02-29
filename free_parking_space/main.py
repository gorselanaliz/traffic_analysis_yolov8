import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import imutils

import numpy as np
from ultralytics import YOLO

blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
black = (0,0,0)
white = (255,255,255)

thickness = 2
font_scale = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

polygon_1_dict = {}
polygon_2_dict = {}
polygon_3_dict = {}
polygon_4_dict = {}
polygon_5_dict = {}
polygon_6_dict = {}

vehicles = [2, 3, 5, 7]


polygon_1 = np.array([[634, 471],[629, 525],
                      [767, 523],[739, 465]], np.int32)

polygon_2 = np.array([[752, 464],[785, 521],
                      [879, 516],[820, 451]], np.int32)

polygon_3 = np.array([[897, 516],[837, 450],
                      [886, 441],[968, 507]], np.int32) 

polygon_4 = np.array([[981, 505],[898, 437],
                      [954, 430],[1049, 497]], np.int32)

polygon_5 = np.array([[1060, 493],[965, 425],
                      [1018, 421],[1118, 483]], np.int32)

polygon_6 = np.array([[1127, 481],[1030, 417],
                       [1093, 408],[1181, 467]], np.int32)


video_path = "inference/test.mp4"
cap = cv2.VideoCapture(video_path)
model = YOLO("models/yolov8n.pt")

width = 1280
height = 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("video.avi", fourcc, 20.0, (width, height))

free_space_counter = []

while True:
    ret, frame = cap.read()
    if ret==False:
        break
    
    frame = imutils.resize(frame, width=1280)
    frame_copy = frame.copy()
    
    cv2.polylines(frame_copy, [polygon_1], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_2], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_3], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_4], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_5], isClosed=True, color=green, thickness=thickness)
    cv2.polylines(frame_copy, [polygon_6], isClosed=True, color=green, thickness=thickness)
    
    results = model.track(frame, persist=True, verbose=False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype="int")
    
    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        
        if class_id in  vehicles:
            cv2.circle(frame_copy, (cx, cy), 3, blue, -1)
            cv2.rectangle(frame_copy, (x1,y1), (x2,y2), blue, thickness=1) 
            
            polygon_1_result = cv2.pointPolygonTest(polygon_1, (cx, cy), measureDist=False)
            polygon_2_result = cv2.pointPolygonTest(polygon_2, (cx, cy), measureDist=False)
            polygon_3_result = cv2.pointPolygonTest(polygon_3, (cx, cy), measureDist=False)
            polygon_4_result = cv2.pointPolygonTest(polygon_4, (cx, cy), measureDist=False)
            polygon_5_result = cv2.pointPolygonTest(polygon_5, (cx, cy), measureDist=False)
            polygon_6_result = cv2.pointPolygonTest(polygon_6, (cx, cy), measureDist=False)
            
            if polygon_1_result >= 0:
                # print("Polygon 1!")
                polygon_1_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_1_dict)
                # cv2.putText(frame_copy, "Not Free", (cx,cy), font, 0.8, red, thickness)
                
            if polygon_2_result >= 0:
                # print("Polygon 2!")
                polygon_2_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_2_dict)
                # cv2.putText(frame_copy, "Not Free", (cx,cy), font, 0.8, red, thickness)
                
            if polygon_3_result >= 0:
                # print("Polygon 3!")
                polygon_3_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_3_dict)
                # cv2.putText(frame_copy, "Not Free", (cx,cy), font, 0.8, red, thickness)
                
            if polygon_4_result >= 0:
                # print("Polygon 4!")
                polygon_4_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_4_dict)
                # cv2.putText(frame_copy, "Not Free", (cx,cy), font, 0.8, red, thickness)
                
            if polygon_5_result >= 0:
                # print("Polygon 5!")
                polygon_5_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_5_dict)
                # cv2.putText(frame_copy, "Not Free", (cx,cy), font, 0.8, red, thickness)
                
            if polygon_6_result >= 0:
                # print("Polygon 6!")
                polygon_6_dict[track_id] = x1, y1, x2, y2
                free_space_counter.append(polygon_6_dict)
                # cv2.putText(frame_copy, "Not Free", (cx,cy), font, 0.8, red, thickness)
    
     
    free_lot = 6-len(free_space_counter)
    free_space_counter.clear()
    
    text = "Free Parking Lot: {}".format(free_lot)
    cv2.putText(frame_copy, text, (10, 25), font, 0.8, black, thickness)
    
    writer.write(frame_copy)
    cv2.imshow("Free Parking Space Counter", frame_copy)
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break
    
cap.release()
writer.release()

cv2.destroyAllWindows()
print("[INFO]...Processing is finished!")
