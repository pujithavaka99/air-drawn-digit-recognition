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
- Gesture-based interaction without keyboard controls

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
````

## Image Preprocessing

The drawn digit undergoes the following preprocessing steps before classification:

1. Convert the drawing to grayscale.
2. Identify the region containing the digit.
3. Crop the digit.
4. Convert the image into a square format.
5. Resize the image to 28 × 28 pixels.
6. Normalize pixel values.
7. Flatten the image into 784 features.
8. Pass the processed image to the KNN classifier.

## Machine Learning Model

The project uses a K-Nearest Neighbors classifier for digit recognition.

### Model Configuration

```text
Algorithm: KNeighborsClassifier
Number of Neighbors (K): 3
Weights: distance
Classes: 0–9
Input Features: 784
```

The selected KNN model achieved an accuracy of 96.61% on the test dataset.

## Model Performance

The dataset contains 40,430 grayscale digit images.

The final KNN model achieved:

```text
Accuracy: 96.61%
```

## Gesture Controls

| Gesture             | Action                   |
| ------------------- | ------------------------ |
| Thumb + Index Pinch | Start drawing            |
| Release Pinch       | Stop drawing and predict |
| Open Palm           | Clear canvas             |
| Two Fingers         | Exit                     |

No keyboard input is required during the application.

## Project Structure

```text
air-drawn-digit-recognition/
│
├── app.py
├── trained_model.pkl
├── requirements.txt
└── README.md
```

The training dataset is not required for inference when the trained model is already available as `trained_model.pkl`.

## Installation

Clone the repository:

```bash
git clone https://github.com/pujithavaka99/air-drawn-digit-recognition.git
```

Navigate to the project directory:

```bash
cd air-drawn-digit-recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the application using:

```bash
python app.py
```

Allow webcam access when prompted.

The application will detect the user's hand, track the finger movement, display the air-drawn digit, and predict the corresponding digit.

## Dataset

The model was trained using a grayscale handwritten digit dataset containing 40,430 images.

Each image is processed into a 28 × 28 representation and flattened into 784 numerical features before being provided to the machine learning model.

## Challenges

The main challenges addressed in the project include:

* Detecting hand movements accurately in real time
* Converting finger movements into a clean digit drawing
* Handling different drawing sizes and positions
* Converting the drawn image into the format expected by the model
* Improving classification performance through model selection and tuning
* Maintaining real-time performance during webcam processing

## Future Improvements

* Replace KNN with a CNN-based digit recognition model
* Improve recognition of visually similar digits
* Add prediction history
* Improve drawing stabilization
* Support handwritten alphabet recognition
* Add browser-based deployment
* Improve real-time prediction speed

## Author

Pujitha Vaka

GitHub: [https://github.com/pujithavaka99](https://github.com/pujithavaka99)

```
```
