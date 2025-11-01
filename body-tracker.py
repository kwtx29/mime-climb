#Required things
import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import overlay_lib
from overlay_lib import Vector2D, RgbaColor, SkDrawCircle

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def build_overlay_items_from_results(image, results):
    """Return a list of overlay items (lines + circles) for the current frame."""
    overlay_items = []
    h, w = image.shape[:2]
    if not results or not results.multi_hand_landmarks:
        return overlay_items

    for hl in results.multi_hand_landmarks:
        # Draw skeleton connections
        for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
            s = hl.landmark[start_idx]
            e = hl.landmark[end_idx]
            sx, sy = int(s.x * w), int(s.y * h)
            ex, ey = int(e.x * w), int(e.y * h)
            overlay_items.append(
                overlay_lib.DrawLine(
                    Vector2D(sx, sy),
                    Vector2D(ex, ey),
                    RgbaColor(0, 255, 0, 255),
                    2
                )
            )
        # Draw landmark circles
        for lm in hl.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            overlay_items.append(
                SkDrawCircle(Vector2D(x, y), 6, RgbaColor(255, 255, 255, 255), 2)
            )
    return overlay_items

def callback():
    """Overlay drawlist callback used by overlay_lib. Returns items for the latest camera frame."""
    ret, image = cap.read()
    if not ret:
        return []
    # Flip + convert for mediapipe
    mp_image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(mp_image)
    # Build overlay items from results (use the original BGR image size)
    items = build_overlay_items_from_results(image, results)
    return items

overlay = overlay_lib.Overlay(
    drawlistCallback=callback,
    refreshTimeout=1
)

overlay.spawn()

try:
    while True:
        ret, image = cap.read()
        if not ret:
            break
        # Flip and convert for display / processing
        mp_image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(mp_image)
        # Build overlay items and hand them to the overlay instance for this frame
        overlay_items = build_overlay_items_from_results(image, results)
        overlay.drawlistCallback = lambda items=overlay_items: items

        # Optional: show the camera image locally with MediaPipe landmark drawing for debugging
        display_img = cv2.cvtColor(mp_image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_img,
                    hl,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
                )

        cv2.imshow('Hand Tracking', display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    overlay.terminate()  # if overlay_lib exposes terminate/stop; if not, safe to ignore