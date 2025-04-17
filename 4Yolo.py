# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:26:13 2025

@author: vikas
"""

import cv2
import numpy as np

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detect objects
def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), 
                                mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs

# Process outputs
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
                
                # Rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    return boxes, confidences, class_ids, indices

# Draw bounding boxes
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

# Main function
def main():
    # Load YOLO model
    net, classes, output_layers = load_yolo()
    
    # Load image
    #img = cv2.imread("horses.jpg")
    img = cv2.imread("giraffe.jpg")
    
    # Detect objects
    outputs = detect_objects(img, net, output_layers)
    
    # Process outputs
    boxes, confidences, class_ids, indices = process_outputs(img, outputs, classes)
    
    # Draw results
    result_img = draw_labels(img.copy(), boxes, confidences, class_ids, classes, indices)
    
    # Display output
    cv2.imshow("YOLO Object Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()