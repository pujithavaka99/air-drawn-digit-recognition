# Air-Drawn Digit Recognition

A real-time computer vision application that allows users to draw digits in the air using hand gestures and recognizes the drawn digit using a machine learning model.

The project combines MediaPipe for real-time hand tracking, OpenCV for video processing and virtual drawing, and Scikit-learn for digit classification.

## Project Overview

The system captures hand movements through a webcam and tracks the user's index finger using MediaPipe. The finger movement is converted into a virtual drawing on the screen.

Once the digit is completed, the drawing is processed into a 28 × 28 grayscale image and passed to a trained K-Nearest Neighbors (KNN) classifier for prediction.

The system recognizes digits from 0 to 9.

## Features

- Real-time webcam-based digit recognition
- Hand tracking using MediaPipe
- Air drawing using hand movements
- Virtual drawing canvas using OpenCV
- Recognition of digits 0–9
- Image preprocessing and resizing
- KNN-based digit classification
- Prediction confidence display
- Real-time prediction pipeline

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Joblib
- K-Nearest Neighbors (KNN)

## System Workflow

```text
Webcam
   |
   v
Hand Detection using MediaPipe
   |
   v
Index Finger Tracking
   |
   v
Air Drawing
   |
   v
Virtual Canvas
   |
   v
Image Preprocessing
   |
   v
Grayscale Image
   |
   v
Resize to 28 × 28
   |
   v
Flatten to 784 Features
   |
   v
KNN Classifier
   |
   v
Digit Prediction
