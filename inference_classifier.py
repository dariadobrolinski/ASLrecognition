import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from collections import deque

engine = pyttsx3.init()
engine.setProperty('rate', 150) 
engine.setProperty('volume', 1.0)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

last_spoken = None 
last_spoken_time = 0 
cooldown_time = 1.5

prediction_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            data_aux = []
            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            min_x, min_y = min(x_), min(y_)

            for lm in hand_landmarks.landmark:
                data_aux.extend([lm.x - min_x, lm.y - min_y])

            if len(data_aux) < 42:
                data_aux.extend([0] * (42 - len(data_aux)))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

            prediction_buffer.append(predicted_character)

            if prediction_buffer.count(predicted_character) > 7: 
                current_time = time.time()
                if predicted_character != last_spoken and (current_time - last_spoken_time) > cooldown_time:
                    engine.say(predicted_character)
                    engine.runAndWait()
                    last_spoken = predicted_character
                    last_spoken_time = current_time

            x1 = int(min_x * W) - 10
            y1 = int(min_y * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 20),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('ASL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()