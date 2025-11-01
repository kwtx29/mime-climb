#Required things
import cv2
import mediapipe as mp
import tkinter as tk
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands=mp.solutions.hands

import overlay_lib
from overlay_lib import Vector2D, RgbaColor, SkDrawCircle

overlay_items = []

def callback():
    return overlay_items

overlay = overlay_lib.Overlay(
    drawlistCallback=callback,
    refreshTimeout=1
)

overlay.spawn()




cap=cv2.VideoCapture(0)
hands=mp_hands.Hands()
while True:
    data,image = cap.read()
    # Flip the image
    image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
    # storing the results
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Build overlay draw list for all hands (handle once per frame)
            h, w = image.shape[:2]
            overlay_items = []
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hl, mp_hands.HAND_CONNECTIONS)
                for lm in hl.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    overlay_items.append(
                        SkDrawCircle(Vector2D(x, y), 6, RgbaColor(255, 255, 255, 255), 2)
                    )
            # Update the overlay callback to return the current frame's draw items
            overlay.drawlistCallback = lambda items=overlay_items: items
            # we've handled all hands for this frame, break the outer loop iteration
            break
            
    print(results.multi_hand_landmarks)
    cv2.imshow('Hand Tracking',image)
    cv2.waitKey(1)