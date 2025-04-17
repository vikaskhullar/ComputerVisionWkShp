# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 12:08:30 2025

@author: vikas
"""

# First, ensure you have the required packages:
# pip install opencv-python opencv-contrib-python numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# 1. Basic Image Operations
# --------------------

# Read and display an image
img = cv2.imread('cars.jpg')
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)
cv2.waitKey(0)

# Image blurring (noise reduction)
blurred = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imshow('Blurred Image', blurred)
cv2.waitKey(0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 100, 200)
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------
# 2. Object Detection (Face Detection)
# --------------------

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------
# 4. Video Processing (Real-time Edge Detection)
# --------------------
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Live Edge Detection', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('Live Edge Detection', edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

