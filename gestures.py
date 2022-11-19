import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

import requests
import json

from math import sqrt

mediapipe_hands_obj = mp.solutions.hands
hands_obj = mediapipe_hands_obj.Hands(max_num_hands=1, min_detection_confidence=0.7)
mediapipe_draw_obj = mp.solutions.drawing_utils

model = load_model("static/model/mp_hand_gesture")
gesture_names_file = open("static/model/gesture.names", "r")
class_names = gesture_names_file.read().split("\n")
gesture_names_file.close()


class Hand:

    def __init__(self, hand_model, hand_gesture_model, class_names):
        self.hand_model = hand_model
        self.hand_gesture_model = hand_gesture_model
        self.class_names = class_names


    def get_data(self, frame):
        x, y, c = frame.shape
        return_dict = {
                "className": "",
            }

        result = self.hand_model.process(frame)

        class_name = ""

        list_of_x_coords = []
        list_of_y_coords = []

        if result.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in result.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append([int(landmark.x*x), int(landmark.y*y)])
                    list_of_x_coords.append(int(landmark.x*x)-x/2)
                    list_of_y_coords.append(y/2-int(landmark.y*y))

            prediction = self.hand_gesture_model.predict([landmarks])

            class_id = np.argmax(prediction)
            class_name = self.class_names[class_id]

        try:
            avg_x = sum(list_of_x_coords)/len(list_of_x_coords)
            avg_y = sum(list_of_y_coords)/len(list_of_y_coords)

        except ZeroDivisionError:
            avg_x = None
            avg_y = None

        return_dict["className"] = class_name
        return_dict["direction"] = [avg_x, avg_y]
        try:
            return_dict["magnitude"] = sqrt(avg_x**2+avg_y**2)/sqrt((x)**2+(y)**2)

        except TypeError:
            return_dict["magnitude"] = 0

        print(return_dict)

        return return_dict




capture = cv2.VideoCapture(0)

hand_obj = Hand(hand_model=hands_obj, hand_gesture_model=model, class_names=class_names)

while True:
    _, frame = capture.read()
    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_obj.process(frame_rgb)

    class_name = ""

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:

            for lm in handslms.landmark:
                lmx = int(lm.x*x)
                lmy = int(lm.y*y)

                landmarks.append([lmx, lmy])
            mediapipe_draw_obj.draw_landmarks(frame, handslms, mediapipe_hands_obj.HAND_CONNECTIONS)
            data_dict = hand_obj.get_data(frame)
            print(data_dict)
        print(data_dict)
        cv2.putText(frame, data_dict["className"], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("robotic arm control", frame)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
