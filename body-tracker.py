#Required things
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import overlay_lib
from overlay_lib import Vector2D, RgbaColor, SkDrawCircle, FlDrawCircle

cap = cv2.VideoCapture(0)

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)


body_x = 0
body_y = 0


def actual_to_relative_coords(x, y, h,w, scale=0.3):
    x = (x-w/2-body_x)*scale
    y = -1*(y-h+body_y)*scale
    return x,y

def relative_to_actual_coords(x, y, h, w):
    x = x + w/2 + body_x
    y = -1*y + h - body_y
    return int(x),  int(y)



def update_body_position(LhandPos, RhandPos, h,w):
    global body_x, body_y
    Lx, Ly = LhandPos.x*w*0.3, LhandPos.y*h*0.3
    Rx, Ry = RhandPos.x*w*0.3, RhandPos.y*h*0.3
    body_x = w*0.3 - (Lx + Rx) / 2
    body_y = (Ly + Ry) / 2




def build_overlay_items_from_results(results):
    """Return a list of overlay items (lines + circles) for the current frame."""
    overlay_items = []
    h, w = 1080, 1920
    if not results or not results.hand_landmarks:
        return overlay_items
    LRpos = []
    for hl in results.hand_landmarks:
        LRpos += [hl[0]]
        # Draw skeleton connections
        for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
            s = hl[start_idx]
            e = hl[end_idx]
            sx, sy = actual_to_relative_coords(s.x * w, s.y * h, h, w)
            ex, ey = actual_to_relative_coords(e.x * w, e.y * h, h, w)
            sx, sy = relative_to_actual_coords(sx, sy, h, w)
            ex, ey = relative_to_actual_coords(ex, ey, h, w)
                
            overlay_items.append(
                overlay_lib.DrawLine(
                    Vector2D(sx, sy),
                    Vector2D(ex, ey),
                    RgbaColor(0, 255, 0, 255),
                    12
                )
            )
        # Draw landmark circles
        for lm in hl:
            x,y = actual_to_relative_coords(lm.x * w, lm.y * h, h, w)
            x,y = relative_to_actual_coords(x, y, h, w)

            overlay_items.append(
                FlDrawCircle(Vector2D(x, y), 6, RgbaColor(255, 255, 255, 255), RgbaColor(255, 255, 255, 255), 0)
            )
            

    dragging = True
    if dragging and len(LRpos) == 2:
        update_body_position(LRpos[0], LRpos[1], h, w)

    x, y = relative_to_actual_coords(body_x, body_y, h, w)
    overlay_items.append(FlDrawCircle(Vector2D(x,y), 50, RgbaColor(255, 0, 0, 128), RgbaColor(255, 0, 0, 255), 2))
    return overlay_items

def callback():
    """Overlay drawlist callback used by overlay_lib. Returns items for the latest camera frame."""
    ret, frame = cap.read()
    if not ret:
        return []
    # Flip + convert for mediapipe
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = recognizer.recognize(image)
    # Build overlay items from results (use the original BGR image size)
    items = build_overlay_items_from_results(results)
    return items

overlay = overlay_lib.Overlay(
    drawlistCallback=callback,
    refreshTimeout=1
)

overlay.spawn()

