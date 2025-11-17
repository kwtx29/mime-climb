#nuitka setup
#nuitka-project: --mode=onefile
#nuitka-project: --windows-console-mode=disable
#nuitka-project: --include-data-files={MAIN_DIRECTORY}/body_marker.png=body_marker.png
#nuitka-project: --include-data-files={MAIN_DIRECTORY}/gesture_recognizer.task=gesture_recognizer.task
#nuitka-project: --enable-plugin=pyqt5

#^this is shit

#Required things
import cv2
import mediapipe as mp
import threading
import time
import pyautogui as pg
import overlay_lib
from overlay_lib import Vector2D, RgbaColor, SkDrawCircle, FlDrawCircle, DrawImage, Size2D, DrawText
from gesture_recognizer import get_gesture, get_gesture_helper
from typing import List, Optional, Tuple

# Constants
Z_THRESHOLD = -0.12
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SCALE = 0.2

# MediaPipe & gesture recognizer init (created once)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = tuple(mp_hands.HAND_CONNECTIONS)






cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    


# Prefer a smaller camera resolution and FPS to reduce CPU/GPU load
try:
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    #^reduction of accuracy to save resources

    # Reduce internal buffering to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Many webcams are faster with MJPG
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
except Exception:
    pass


# NOTE: Gesture recognition is handled in gesture_recognizer.py via LIVE_STREAM callback.


# Shared storage for latest overlay items produced by the processing thread
_latest_items_lock = threading.Lock()
_latest_overlay_items = []

_running = True











class Tracker:
    def __init__(self):
        # body position (screen coords) and smoothed versions
        self.body_x = SCREEN_WIDTH // 2
        self.body_y = SCREEN_HEIGHT // 2
        self.smoothed_body_x = float(self.body_x)
        self.smoothed_body_y = float(self.body_y)

        self.prior_gesture = None
        self.prior_z_bin = False
        self.prior_swipe_start: Optional[List[Tuple[float, float]]] = None
        self.prior_hand_num = 0
        self.prior_overlay_results: List = []

        # smoothing factor (0..1). Lower => smoother/slower
        self.smooth_alpha = 0.45

    @staticmethod
    def average(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def get_hands(positions: List[Tuple[float, float]]):
        if not positions:
            return None, None
        if len(positions) == 2:
            positions_sorted = sorted(positions, key=lambda p: p[0])
            return positions_sorted[0], positions_sorted[1]
        p = positions[0]
        if p[0] < 0.5:
            return p, None
        return None, p

    @staticmethod
    def relative_to_actual_coords(x: float, y: float) -> Tuple[int, int]:
        return int(x * SCREEN_WIDTH), int(y * SCREEN_HEIGHT)

    def actual_to_around_body_coords(self, x: int, y: int) -> Tuple[int, int]:
        x_diff = (x - SCREEN_WIDTH // 2) * SCALE
        y_diff = (y - SCREEN_HEIGHT // 2) * SCALE
        return int(self.body_x + x_diff), int(self.body_y + y_diff)
    
    @staticmethod
    def _click_at(x: int, y: int):
        pg.click(x=x, y=y, button='left')


    def update_body_position(self, left: Tuple[int, int], right: Tuple[int, int]):
        if not left or not right:
            return
        x_a_l, y_a_l = left
        x_a_r, y_a_r = right
        new_x = int((x_a_l + x_a_r) / 2)
        new_y = int((y_a_l + y_a_r) / 2)

        # exponential smoothing to make motion smoother
        self.smoothed_body_x = (self.smooth_alpha * new_x) + ((1.0 - self.smooth_alpha) * self.smoothed_body_x)
        self.smoothed_body_y = (self.smooth_alpha * new_y) + ((1.0 - self.smooth_alpha) * self.smoothed_body_y)

        # expose integer body_x/body_y for downstream uses (clicks/draw)
        self.body_x = int(self.smoothed_body_x)
        self.body_y = int(self.smoothed_body_y)

    def build_overlay_items(self):
        # If recognizer returns None or empty, reuse prior results

        hands = get_gesture(h=SCREEN_HEIGHT, w=SCREEN_WIDTH)
        if not hands:
            return self.prior_overlay_results

        overlay_items = []
        wrist_positions: List[Tuple[float, float]] = []
        pointer_z_values: List[float] = []
        gestures: List[str] = []

        # local bindings for speed
        rel_to_actual = self.relative_to_actual_coords
        actual_to_body = self.actual_to_around_body_coords
        DrawLine = getattr(overlay_lib, "DrawLine", None)
        FlCircle = FlDrawCircle
        Vec = Vector2D
        Rgba = RgbaColor

        # Draw body marker
        overlay_items.append(
            DrawImage(
                Vector2D(self.body_x, self.body_y - 32),
                'body_marker.png',
                Size2D(100, 190),
                Vector2D(0, 0)
        ))
        
        
        for hand in hands:
            relative_landmarks = hand.get('relative_landmarks', [])
            if not relative_landmarks:
                continue

            wrist_x, wrist_y, _ = relative_landmarks[0]
            wrist_positions.append((wrist_x, wrist_y))

            px, py, pz = relative_landmarks[8]  # index finger tip
            px, py = rel_to_actual(px, py)
            px, py = actual_to_body(px, py)
            pointer_z_values.append([px, py, pz])

            gestures.append(hand.get('gesture'))

            n_landmarks = len(relative_landmarks)
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx >= n_landmarks or end_idx >= n_landmarks:
                    continue
                x_r_s, y_r_s, _ = relative_landmarks[start_idx]
                x_r_e, y_r_e, _ = relative_landmarks[end_idx]

                x_a_s, y_a_s = rel_to_actual(x_r_s, y_r_s)
                x_a_e, y_a_e = rel_to_actual(x_r_e, y_r_e)

                x_l_s, y_l_s = actual_to_body(x_a_s, y_a_s)
                x_l_e, y_l_e = actual_to_body(x_a_e, y_a_e)

                if DrawLine:
                    overlay_items.append(
                        DrawLine(Vec(x_l_s, y_l_s), Vec(x_l_e, y_l_e), Rgba(143, 229, 152, 220), 6)
                    )

            # Draw landmark circles (reduced radius)
            for (x_r, y_r, _) in relative_landmarks:
                x_a, y_a = rel_to_actual(x_r, y_r)
                x_l, y_l = actual_to_body(x_a, y_a)
                overlay_items.append(
                        FlCircle(Vec(x_l, y_l), 3, Rgba(255, 255, 255, 220), Rgba(143, 229, 152, 220), 0)
                    )


        # Click handling using pointer Z
        if gestures and any(point_hand := [g == 'Pointing_Up' for g in gestures]):
            point_hand = point_hand.index(True)
            z_min = min(i for _, _, i in pointer_z_values) if pointer_z_values else 1.0
            z_bin = z_min <= Z_THRESHOLD
            if z_bin and (z_bin != self.prior_z_bin):
                px, py, _ = pointer_z_values[point_hand]
                threading.Thread(target=self._click_at, args=(px, py), daemon=True).start()

                

            self.prior_z_bin = z_bin
        else:
            self.prior_z_bin = False

        # Swipe handling (keeps behaviour but uses smoothed coords)
        if self.prior_swipe_start is not None:
            new_left, new_right = self.get_hands(wrist_positions) if wrist_positions else (None, None)
            old_left, old_right = self.get_hands(self.prior_swipe_start)

            x_diffs = []
            y_diffs = []

            if new_left and old_left:
                x_diffs.append(new_left[0] - old_left[0])
                y_diffs.append(new_left[1] - old_left[1])
            if new_right and old_right:
                x_diffs.append(new_right[0] - old_right[0])
                y_diffs.append(new_right[1] - old_right[1])

            if x_diffs and y_diffs:
                x_r = self.average(x_diffs)
                y_r = self.average(y_diffs)
                x_a, y_a = self.relative_to_actual_coords(x_r, y_r)
                # apply smaller increments to avoid jumps
                self.body_x += int(x_a * -0.4)
                self.body_y += int(y_a * -0.4)

            self.prior_swipe_start = None
        else:
            if gestures and (n := [g == 'Closed_Fist' for g in gestures].count(True)): #checks if any hand is making a fist
                self.prior_hand_num = n

            
                self.prior_swipe_start = wrist_positions.copy()



        self.prior_overlay_results = overlay_items
        return overlay_items


tracker = Tracker()


def _process_loop():
    """Background processing thread: consumes the latest frame, runs recognizer, produces overlay items."""
    global _latest_overlay_items, _running
    while _running:
        try:
            items = tracker.build_overlay_items()
            # publish latest items
            with _latest_items_lock:
                _latest_overlay_items = items
        except Exception:
            continue
        # Throttle processing to avoid pegging a CPU core
        time.sleep(0.008)


def _camera_loop():
    """Background thread for reading frames from the camera to avoid blocking."""
    global _running
    while _running:
        ret, frame = cap.read()
        if not ret:
            _running = False
            break
        frame = cv2.flip(frame, 1)
        get_gesture_helper(frame)
        # Yield to other threads, target ~60-100 Hz camera polling
        time.sleep(0.008)


# start background processor
_processor_thread = threading.Thread(target=_process_loop, daemon=True)
_processor_thread.start()

# start background camera reader
_camera_thread = threading.Thread(target=_camera_loop, daemon=True)
_camera_thread.start()


def callback():
    """Overlay drawlist callback used by overlay_lib. Returns items for the latest camera frame."""
    with _latest_items_lock:
        items_to_draw = list(_latest_overlay_items)
    
    return items_to_draw


overlay = overlay_lib.Overlay(drawlistCallback=callback, refreshTimeout=16) # Target ~60fps

# The main thread will now be taken over by the overlay.
# This is the standard pattern for most GUI libraries.
try:
    print("Starting overlay. This will block the main thread. Press Ctrl+C in terminal to exit.")
    overlay.spawn()

except KeyboardInterrupt:
    print("Keyboard interrupt received, shutting down.")
    pass
finally:
    # shutdown all threads
    _running = False
    print("Shutting down threads...")
    if _camera_thread.is_alive():
        _camera_thread.join(timeout=1.0)
    if _processor_thread.is_alive():
        _processor_thread.join(timeout=1.0)
    
    cap.release()
    # overlay.terminate() is likely not needed if spawn() exits cleanly, but good practice
    try:
        overlay.terminate()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("Shutdown complete.")

