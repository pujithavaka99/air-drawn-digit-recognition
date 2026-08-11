# ✋ Air-Drawn Digit Recognition

A real-time computer vision application that converts webcam-tracked hand movements into digit drawings and recognizes digits **0–9** using a trained K-Nearest Neighbors (KNN) classifier. The project combines **MediaPipe** for hand tracking with **OpenCV** for live video processing and drawing.

## 🚀 Features

- Real-time webcam-based digit recognition
- MediaPipe hand tracking
- Air drawing using hand gestures
- 28×28 image preprocessing
- 784-feature KNN input
- Prediction confidence display
- Recognition of digits **0–9**

## 🧠 Machine Learning Model

**Algorithm:** K-Nearest Neighbors (KNN)

```text
K = 3
Weights = distance
Classes = 0–9
Input features = 784
Accuracy = 96.61%
```

The model was trained and tuned on **40,430 grayscale digit images**. Hyperparameter tuning with `n_neighbors=3` and `weights='distance'` improved accuracy from **96.49% to 96.61%**. fileciteturn31file4L217-L221

## 🔄 Workflow

```text
Webcam
   ↓
MediaPipe Hand Tracking
   ↓
Hand Movement Detection
   ↓
Air Drawing
   ↓
Virtual Canvas
   ↓
Image Preprocessing
   ↓
28 × 28 Grayscale Image
   ↓
Flatten to 784 Features
   ↓
KNN Classifier
   ↓
Digit Prediction
```

## 🛠️ Technologies

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Joblib
- K-Nearest Neighbors (KNN)
- Decision Tree

The project uses OpenCV and MediaPipe for the real-time computer vision pipeline and compares KNN with Decision Tree classification. fileciteturn31file2L125-L129

## 📂 Project Structure

```text
air-drawn-digit-recognition/
│
├── app.py
├── trained_model.pkl
├── requirements.txt
└── README.md
```

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

## 🖼️ Image Preprocessing

The air-drawn digit is converted into a grayscale image, resized to **28×28 pixels**, and flattened into **784 features** before prediction. fileciteturn31file15L545-L548

```text
Air-drawn digit
      ↓
Grayscale
      ↓
Crop / prepare image
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

| Model | Result |
|---|---:|
| KNN - Initial | 96.49% |
| KNN - Tuned | **96.61%** |

The tuned KNN model was integrated into the live recognition pipeline. fileciteturn31file4L217-L221

## 🔮 Future Improvements

- Improve robustness for visually similar digits
- Explore CNN-based digit recognition
- Add prediction history
- Improve gesture controls
- Extend the system to handwritten alphabet recognition
- Build a browser-based deployment

## 👩‍💻 Author

**Pujitha Vaka**

GitHub: https://github.com/pujithavaka99
