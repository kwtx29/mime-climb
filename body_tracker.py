#Required things
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import overlay_lib
from overlay_lib import Vector2D, RgbaColor, SkDrawCircle, FlDrawCircle
from gesture_recognizer import get_gesture
import math
import pyautogui as pg


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)

Z_THRESHOLD = -0.15
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCALE = 0.3

class Tracker:
    def __init__(self):
        
        self.body_x = SCREEN_WIDTH // 2
        self.body_y = SCREEN_HEIGHT // 2

        self.prior_gesture = None

        # Z handling (for clicks)
        self.prior_z_bin = False

        # Swipe handling (for movement)
        self.prior_swipe_start = None

        self.prior_overlay_results = []

    @staticmethod
    def average(x):
        return sum(x) / len(x)


    @staticmethod
    def get_hands(positions):

        if len(positions) == 2:
            positions.sort(key = lambda x:x[0])
            left = positions[0]
            right = positions[1]
        else:

            if positions[0][0] < SCREEN_WIDTH // 2:
                left = positions[0]
                right = None
            else:
                left = None
                right = positions[0]

        return left, right


    def relative_to_actual_coords(self, x, y):
        x *= SCREEN_WIDTH
        y *= SCREEN_HEIGHT

        return int(x),  int(y)
    
    def actual_to_around_body_coords(self, x, y):

        # Initially we are in the space [0, w] and [0, h]
        # We want to compare to the midpoint of the screen and collect that vector, 
        # scale it, and apply it to the body


        # e.g. if we are at width 1000 and the screen is of width 1920, we get a difference of 40
        x_diff = x - SCREEN_WIDTH//2
        y_diff = y - SCREEN_HEIGHT//2

        x_diff *= SCALE
        y_diff *= SCALE

        x_body = self.body_x + x_diff
        y_body = self.body_y + y_diff

        return int(x_body), int(y_body)




    def update_body_position(self, left, right):

        x_r_l, y_r_l = left
        x_r_r, y_r_r = right

        x_a_l, y_a_l = self.relative_to_actual_coords(x_r_l, y_r_l)
        x_a_r, y_a_r = self.relative_to_actual_coords(x_r_r, y_r_r)

        self.body_x = (x_a_l + x_a_r) / 2
        self.body_y = (y_a_l + y_a_r) / 2
        

    def build_overlay_items_from_results(self, results):
        """Return a list of overlay items (lines + circles) for the current frame."""

        
        if not results: return self.prior_overlay_results
        overlay_items = []

        hands = get_gesture(results, 
                            h=SCREEN_HEIGHT, 
                            w=SCREEN_HEIGHT)
        if not hands: return self.prior_overlay_results

        wrist_positions = []
        pointer_z_values = []
        gesture = []

        for hand in hands:

            relative_landmarks = hand['relative_landmarks']

            x, y, z = relative_landmarks[0] # First landmark is wrist, documented
            wrist_positions.append((x, y)) 

            x, y, z = relative_landmarks[-1] # Last landmark is pointer finger (?)
            pointer_z_values.append(z)

            gesture.append(hand['gesture'])

            # Draw skeleton connections
            for start_idx, end_idx in mp_hands.HAND_CONNECTIONS: 
                x_r_s, y_r_s, z = relative_landmarks[start_idx]
                x_r_e, y_r_e, z = relative_landmarks[end_idx]

                x_a_s, y_a_s = self.relative_to_actual_coords(x_r_s, y_r_s)
                x_a_e, y_a_e = self.relative_to_actual_coords(x_r_e, y_r_e)

                x_l_s, y_l_s = self.actual_to_around_body_coords(x_a_s, y_a_s)
                x_l_e, y_l_e = self.actual_to_around_body_coords(x_a_e, y_a_e)
                    
                overlay_items.append(
                    overlay_lib.DrawLine(
                        Vector2D(x_l_s, y_l_s),
                        Vector2D(x_l_e, y_l_e),
                        RgbaColor(0, 255, 0, 255),
                        12
                    )
                )
            # Draw landmark circles
            for landmark in relative_landmarks:
                x_r, y_r, z = landmark
                x_a, y_a = self.relative_to_actual_coords(x_r, y_r)
                x_l, y_l = self.actual_to_around_body_coords(x_a, y_a)

                overlay_items.append(
                    FlDrawCircle(Vector2D(x_l, y_l), 3, RgbaColor(255, 255, 255, 255), RgbaColor(255, 255, 255, 255), 0)
                )

        dragging = False
        if dragging and len(wrist_positions) == 2:
            self.update_body_position(wrist_positions[0], wrist_positions[1])

        # Click handling

        z_bin = min(pointer_z_values) <= Z_THRESHOLD
        if z_bin != self.prior_z_bin:

            if z_bin: # We are below the threshold, so we point

                print("CLICK")
                pg.click(self.body_x, self.body_y, button='left')

            self.prior_z_bin = z_bin


        # Handling swipes
        # We have a closed fist detected and we have a swipe in progress

        if self.prior_swipe_start is not None:

            new_left, new_right = self.get_hands(wrist_positions)

            old_left, old_right = self.get_hands(self.prior_swipe_start)

            x_diffs = []
            y_diffs = []

            if new_left and old_left:
                x_diffs.append(new_left[0] - old_left[0])
                y_diffs.append(new_left[1] - old_left[1])

            if new_right and old_right:
                x_diffs.append(new_right[0] - old_right[0])
                y_diffs.append(new_right[1] - old_right[1])

            x_r, y_r = self.average(x_diffs), self.average(y_diffs)
            x_a, y_a = self.relative_to_actual_coords(x_r, y_r)

            # Now, we project this onto the body
            self.body_x += int(x_a * 1.2)
            self.body_y += int(y_a * 1.2) # scale factor to easily move about

            self.prior_swipe_start = None

        elif all(g == 'Closed_Fist' for g in gesture):
            self.prior_swipe_start = wrist_positions

        overlay_items.append(FlDrawCircle(Vector2D(self.body_x, self.body_y), 50, RgbaColor(255, 0, 0, 128), RgbaColor(255, 0, 0, 255), 2))
        
        self.prior_overlay_results = overlay_items

        return overlay_items

tracker = Tracker()

def callback():
    """Overlay drawlist callback used by overlay_lib. Returns items for the latest camera frame."""
    ret, frame = cap.read()
    if not ret:
        return []
    
    frame = cv2.flip(frame, 1)

    # Flip + convert for mediapipe
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = recognizer.recognize(image)
    # Build overlay items from results (use the original BGR image size)
    items = tracker.build_overlay_items_from_results(results)

    return items

overlay = overlay_lib.Overlay(
    drawlistCallback=callback,
    refreshTimeout=1
)

overlay.spawn()

