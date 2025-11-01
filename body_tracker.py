#Required things
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import overlay_lib
from overlay_lib import Vector2D, RgbaColor, SkDrawCircle, FlDrawCircle
from gesture_recognizer import get_gesture


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)

class Tracker:
    def __init__(self, 
                 screen_width=1920, 
                 screen_height=1080):
        
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.body_x = screen_width // 2
        self.body_y = screen_height // 2

        self.prior_gesture = None


    def relative_to_actual_coords(self, x, y):
        x *= self.screen_width
        y *= self.screen_height

        return int(x),  int(y)
    
    def actual_to_around_body_coords(self, x, y):

        # Initially we are in the space [0, w] and [0, h]
        # We want to compare to the midpoint of the screen and collect that vector, 
        # scale it, and apply it to the body

        scale = 0.3

        # e.g. if we are at width 1000 and the screen is of width 1920, we get a difference of 40
        x_diff = x - self.screen_width//2
        y_diff = y - self.screen_height//2

        x_diff *= scale
        y_diff *= scale

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

        
        if not results: return []
        overlay_items = []

        hands = get_gesture(results, 
                            h=self.screen_height, 
                            w=self.screen_height)
        if not hands: return []

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
                    FlDrawCircle(Vector2D(x_l, y_l), 6, RgbaColor(255, 255, 255, 255), RgbaColor(255, 255, 255, 255), 0)
                )

        dragging = False
        if dragging and len(wrist_positions) == 2:
            self.update_body_position(wrist_positions[0], wrist_positions[1])

        if gesture != self.prior_gesture:

            print(gesture)

            self.prior_gesture = gesture

    
        overlay_items.append(FlDrawCircle(Vector2D(self.body_x, self.body_y), 50, RgbaColor(255, 0, 0, 128), RgbaColor(255, 0, 0, 255), 2))
        
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

