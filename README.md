# ✋ Air-Drawn Digit Recognition

A real-time digit recognition system that allows users to draw digits **0–9 in the air using hand gestures**. The system uses **MediaPipe Hands** for hand tracking, **OpenCV** for the virtual drawing canvas and image processing, and a trained **K-Nearest Neighbors (KNN)** model for digit classification.

## 🚀 Features

- Real-time webcam-based digit recognition
- MediaPipe hand tracking
- Air drawing using hand gestures
- Pinch gesture to start drawing
- Release pinch to predict
- Open palm gesture to clear the canvas
- Two-finger gesture to exit
- Recognition of digits **0–9**
- 28×28 image preprocessing
- 784-feature KNN input
- Prediction confidence display
- No keyboard controls

## 🧠 Machine Learning Model

**Algorithm:** K-Nearest Neighbors (KNN)

```text
K = 3
Weights = distance
Classes = 0–9
Input features = 784
Test accuracy = 96.52%
```

The trained model is stored as `trained_model.pkl` and loaded during application startup.

## 🔄 Workflow

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
Index Finger Tracking
   ↓
Pinch Gesture
   ↓
Air Drawing
   ↓
Virtual Canvas
   ↓
Image Preprocessing
   ↓
28 × 28 Grayscale Image
   ↓
784 Features
   ↓
KNN Model
   ↓
Digit Prediction
   ↓
Confidence Score
```

## 🛠️ Technologies

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Joblib
- KNN
- VS Code / Jupyter Notebook

## 📂 Project Structure

```text
air-drawn-digit-recognition/
│
├── app.py
├── trained_model.pkl
├── requirements.txt
└── README.md
```

The original training dataset is not required for inference when the trained model is already available.

## ⚙️ Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Run Locally

```bash
python app.py
```

Allow webcam access when prompted.

## 🎮 Gesture Controls

| Gesture | Action |
|---|---|
| 🤏 Thumb + Index Pinch | Start drawing |
| Release Pinch | Stop drawing and predict |
| ✋ Open Palm | Clear canvas |
| ✌️ Two Fingers | Exit |

## 🖼️ Preprocessing

The drawn digit is transformed into a 28×28 image, normalized to the 0–1 range, and flattened into 784 features before being passed to the KNN classifier.

```text
Air-drawn digit
      ↓
Grayscale
      ↓
Crop digit
      ↓
Square image
      ↓
Resize to 28×28
      ↓
Normalize
      ↓
Flatten to 784
      ↓
KNN prediction
```

## 📊 Model Performance

```text
Model        : KNeighborsClassifier
K            : 3
Weights      : distance
Classes      : 0–9
Features     : 784
Accuracy     : 96.52%
```

## 🔮 Future Improvements

- Improve robustness for visually similar digits
- Add CNN-based digit recognition
- Add prediction history
- Improve gesture controls
- Extend the system to handwritten alphabet recognition
- Build a browser-based deployment

## 👩‍💻 Author

**Pujitha Vaka**

GitHub: https://github.com/pujithavaka99
