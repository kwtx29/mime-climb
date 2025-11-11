import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import GestureRecognizerResult
import cv2
import threading
import time
import numpy as np

# Global variables to hold the latest frame and results, with a lock
_latest_result = None
_frame_lock = threading.Lock()

def _result_callback(GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback to receive gesture recognition results."""
    global _latest_result
    with _frame_lock:
        _latest_result = GestureRecognizerResult

# STEP 2: Create a GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=_result_callback
)
recognizer = vision.GestureRecognizer.create_from_options(options)



def get_gesture(result, h, w, frame=None):

    hands = []

    if result.hand_landmarks:

        # iterate hands
        for i, hand in enumerate(result.hand_landmarks):

            hand_data = {
                'relative_landmarks': list((lm.x, lm.y, lm.z) for lm in hand),
                'gesture': None,
            }

            if result.gestures and len(result.gestures) > i and result.gestures[i]:
                gesture = result.gestures[i][0].category_name

                hand_data['gesture'] = gesture

            if frame is not None:

                # draw each landmark on that hand
                for x, y, z in hand_data['relative_landmarks']:
                    px = int(x * w)
                    py = int(y * h)
                    cv2.circle(frame, (px,py), 4, (0,255,0), -1)

                    # use first landmark as anchor text point
                    if hand_data['gesture']:

                        first = hand[0]
                        cv2.putText(frame, gesture, (int(first.x*w), int(first.y*h)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    

            hands.append(hand_data)

        return hands
                    
def draw_gestures_on_frame(frame, result):
    """Draws the recognized gestures and landmarks on the frame."""
    if not result or not result.hand_landmarks:
        return

    h, w, _ = frame.shape
    
    # iterate hands
    for i, hand_landmarks in enumerate(result.hand_landmarks):
        gesture = ''
        if result.gestures and len(result.gestures) > i and result.gestures[i]:
            gesture = result.gestures[i][0].category_name

        # Draw landmarks
        for landmark in hand_landmarks:
            px = int(landmark.x * w)
            py = int(landmark.y * h)
            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        # Draw gesture text
        if gesture:
            first_landmark = hand_landmarks[0]
            cv2.putText(frame, gesture, (int(first_landmark.x * w), int(first_landmark.y * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# Option to run
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    frame_timestamp_ms = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)

        # Convert the frame to a MediaPipe Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Asynchronously recognize gestures.
        frame_timestamp_ms = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        # Draw the latest results on the frame
        with _frame_lock:
            if _latest_result:
                draw_gestures_on_frame(frame, _latest_result)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()