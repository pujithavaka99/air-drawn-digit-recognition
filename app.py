import cv2
import mediapipe as mp
import numpy as np
import joblib


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "trained_model.pkl"

model = joblib.load(MODEL_PATH)

print("========================================")
print("Model loaded successfully!")
print("Classes:", model.classes_)
print("Features:", model.n_features_in_)
print("========================================")


# ============================================================
# MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

print("Camera opened successfully!")


# ============================================================
# SCREEN SETTINGS
# ============================================================

WIDTH = 640
HEIGHT = 480


# ============================================================
# DRAWING AREA
# ============================================================

BOX_X = 120
BOX_Y = 60
BOX_SIZE = 400


# ============================================================
# CANVAS
# ============================================================

canvas = np.zeros(
    (HEIGHT, WIDTH, 3),
    dtype=np.uint8
)


# ============================================================
# DRAWING VARIABLES
# ============================================================

drawing = False
previous_point = None

predicted_digit = None
confidence = None
processed_image = None


# ============================================================
# GESTURE SETTINGS
# ============================================================

PINCH_THRESHOLD = 0.055

clear_counter = 0
exit_counter = 0

CLEAR_FRAMES = 30
EXIT_FRAMES = 30


# ============================================================
# PREPROCESS LIVE DRAWING
# ============================================================

def preprocess_live_image(canvas):

    # --------------------------------------------------------
    # Convert canvas to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Extract only drawing area
    # --------------------------------------------------------

    roi = gray[
        BOX_Y:BOX_Y + BOX_SIZE,
        BOX_X:BOX_X + BOX_SIZE
    ]

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        roi,
        20,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # Check whether canvas is empty
    # --------------------------------------------------------

    if cv2.countNonZero(binary) == 0:
        return None, None

    # --------------------------------------------------------
    # Find digit pixels
    # --------------------------------------------------------

    points = cv2.findNonZero(binary)

    if points is None:
        return None, None

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(points)

    # --------------------------------------------------------
    # Crop digit
    # --------------------------------------------------------

    digit = binary[
        y:y + h,
        x:x + w
    ]

    # --------------------------------------------------------
    # Make image square
    # --------------------------------------------------------

    size = max(w, h)

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = digit

    # --------------------------------------------------------
    # Resize to 28 x 28
    # --------------------------------------------------------

    image_28 = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Center using center of mass
    # --------------------------------------------------------

    moments = cv2.moments(
        image_28
    )

    if moments["m00"] != 0:

        center_x = (
            moments["m10"]
            /
            moments["m00"]
        )

        center_y = (
            moments["m01"]
            /
            moments["m00"]
        )

        shift_x = int(
            round(13.5 - center_x)
        )

        shift_y = int(
            round(13.5 - center_y)
        )

        transformation_matrix = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        image_28 = cv2.warpAffine(
            image_28,
            transformation_matrix,
            (28, 28)
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    processed = (
        image_28.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # Flatten to 784 features
    # --------------------------------------------------------

    features = processed.reshape(
        1,
        784
    )

    return features, image_28


# ============================================================
# PREDICTION
# ============================================================

def predict_digit(canvas):

    features, processed = preprocess_live_image(
        canvas
    )

    if features is None:
        return None, None, None

    # --------------------------------------------------------
    # KNN prediction
    # --------------------------------------------------------

    prediction = model.predict(
        features
    )[0]

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        features
    )

    score = (
        np.max(probabilities)
        * 100
    )

    return (
        prediction,
        score,
        processed
    )


# ============================================================
# FINGER DETECTION
# ============================================================

def finger_up(hand, tip, pip):

    return (
        hand.landmark[tip].y
        <
        hand.landmark[pip].y
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print("Failed to read camera")
        break

    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    frame = cv2.resize(
        frame,
        (WIDTH, HEIGHT)
    )

    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    results = hands.process(
        rgb_frame
    )


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = hand.landmark


        # ====================================================
        # INDEX FINGERTIP
        # ====================================================

        index_tip = landmarks[8]

        x = int(
            index_tip.x
            * WIDTH
        )

        y = int(
            index_tip.y
            * HEIGHT
        )

        current_point = (
            x,
            y
        )

        # Highlight fingertip

        cv2.circle(
            frame,
            current_point,
            10,
            (0, 255, 0),
            cv2.FILLED
        )


        # ====================================================
        # PINCH DETECTION
        # ====================================================

        thumb_tip = landmarks[4]

        pinch_distance = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2
            +
            (thumb_tip.y - index_tip.y) ** 2
        )

        pinching = (
            pinch_distance
            <
            PINCH_THRESHOLD
        )


        # ====================================================
        # FINGER STATES
        # ====================================================

        index_up = finger_up(
            hand,
            8,
            6
        )

        middle_up = finger_up(
            hand,
            12,
            10
        )

        ring_up = finger_up(
            hand,
            16,
            14
        )

        pinky_up = finger_up(
            hand,
            20,
            18
        )


        # ====================================================
        # OPEN PALM → CLEAR
        # ====================================================

        open_palm = (
            index_up
            and middle_up
            and ring_up
            and pinky_up
        )

        if open_palm and not pinching:

            clear_counter += 1

            cv2.putText(
                frame,
                "HOLD OPEN PALM TO CLEAR",
                (140, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            if clear_counter >= CLEAR_FRAMES:

                canvas[:] = 0

                predicted_digit = None
                confidence = None
                processed_image = None

                previous_point = None
                drawing = False

                clear_counter = 0

                print("Canvas cleared")

        else:

            clear_counter = 0


        # ====================================================
        # TWO FINGERS → EXIT
        # ====================================================

        two_fingers = (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
        )

        if (
            two_fingers
            and not pinching
            and not drawing
        ):

            exit_counter += 1

            cv2.putText(
                frame,
                "HOLD TWO FINGERS TO EXIT",
                (150, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            if exit_counter >= EXIT_FRAMES:

                print("Exiting...")
                break

        else:

            exit_counter = 0


        # ====================================================
        # PINCH → DRAW
        # ====================================================

        if pinching:

            inside_box = (
                BOX_X <= x <= BOX_X + BOX_SIZE
                and
                BOX_Y <= y <= BOX_Y + BOX_SIZE
            )

            if inside_box:

                drawing = True

                if previous_point is not None:

                    cv2.line(
                        canvas,
                        previous_point,
                        current_point,
                        (255, 255, 255),
                        18,
                        cv2.LINE_AA
                    )

                previous_point = current_point

                cv2.putText(
                    frame,
                    "DRAWING",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                previous_point = None


        # ====================================================
        # RELEASE PINCH → PREDICT
        # ====================================================

        else:

            if drawing:

                drawing = False

                previous_point = None

                print("Drawing finished...")
                print("Predicting...")

                (
                    predicted_digit,
                    confidence,
                    processed_image
                ) = predict_digit(
                    canvas
                )

                if predicted_digit is not None:

                    print(
                        "Predicted Digit:",
                        predicted_digit
                    )

                    print(
                        f"Confidence: "
                        f"{confidence:.2f}%"
                    )

                    # ----------------------------------------
                    # Save processed image for debugging
                    # ----------------------------------------

                    cv2.imwrite(
                        "live_processed_digit.png",
                        processed_image
                    )

                    # ----------------------------------------
                    # Show 28x28 image
                    # ----------------------------------------

                    preview = cv2.resize(
                        processed_image,
                        (280, 280),
                        interpolation=cv2.INTER_NEAREST
                    )

                    cv2.imshow(
                        "Processed 28x28 Image",
                        preview
                    )


    else:

        previous_point = None


    # ========================================================
    # DRAWING BOX
    # ========================================================

    cv2.rectangle(
        frame,
        (BOX_X, BOX_Y),
        (
            BOX_X + BOX_SIZE,
            BOX_Y + BOX_SIZE
        ),
        (255, 255, 0),
        2
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "AIR-DRAWN DIGIT RECOGNITION",
        (105, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "Pinch = Draw",
        (15, 425),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Open Palm = Clear | Two Fingers = Exit",
        (15, 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # PREDICTION DISPLAY
    # ========================================================

    if predicted_digit is not None:

        cv2.putText(
            frame,
            f"Predicted: {predicted_digit}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}%",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY CAMERA + CANVAS
    # ========================================================

    display_frame = cv2.addWeighted(
        frame,
        1,
        canvas,
        1,
        0
    )

    cv2.imshow(
        "Air Drawn Digit Recognition",
        display_frame
    )

    cv2.imshow(
        "Virtual Canvas",
        canvas
    )

    cv2.waitKey(1)


# ============================================================
# CLEANUP
# ============================================================

cap.release()

hands.close()

cv2.destroyAllWindows()

print("Application closed.")