# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:33:14 2025

@author: vikas
"""

import cv2
import numpy as np
import time

# Load YOLO model (unchanged from previous)
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detection and processing functions (unchanged)
def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), 
                                mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs

def process_outputs(img, outputs, classes, confidence_threshold=0.5, nms_threshold=0.4):
    height, width = img.shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    return boxes, confidences, class_ids, indices

def draw_labels(img, boxes, confidences, class_ids, classes, indices):
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

# Video processing function
def process_video(input_path=0, output_path=None, show_fps=True):
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Load YOLO model
    net, classes, output_layers = load_yolo()
    
    # Initialize performance metrics
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        outputs = detect_objects(frame, net, output_layers)
        boxes, confidences, class_ids, indices = process_outputs(frame, outputs, classes)
        processed_frame = draw_labels(frame.copy(), boxes, confidences, class_ids, classes, indices)
        
        # Calculate and display FPS
        if show_fps:
            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time
            cv2.putText(processed_frame, f"FPS: {current_fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write to output file
        if output_path:
            out.write(processed_frame)
        
        # Display result
        cv2.imshow("YOLO Object Detection", processed_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # For webcam input (device 0):
    # process_video(input_path=0)
    
    # For video file input:
    process_video(input_path="race_car.mp4", output_path="output_video.avi")
    
    # For default webcam with FPS display:
    #process_video(show_fps=True)