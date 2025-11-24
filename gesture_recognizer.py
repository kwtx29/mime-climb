# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import threading
from math import dist


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Thread-safe storage for latest callback output
_latest_lock = threading.Lock()
_latest_result = None 
_latest_frame = None
_busy_lock = threading.Lock()
_busy = False


# Create a gesture recognizer instance with the live stream mode:

def get_result(result, output_image: mp.Image, timestamp_ms: int):
    """Result listener for LIVE_STREAM; stores frame/result for main thread to draw."""
    global _latest_result, _latest_frame, _busy
    # Convert MediaPipe Image (RGB) to numpy BGR for OpenCV drawing
    try:
        frame_rgb = output_image.numpy_view()
    except AttributeError:
        # Fallback for older MP versions
        frame_rgb = output_image.numpy()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Store for main thread display
    with _latest_lock:
        _latest_result = result
        _latest_frame = frame_bgr
    # Mark recognizer as free to accept the next frame
    with _busy_lock:
        _busy = False

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result,
    num_hands=2)



# IMPORTANT: Use the same API family for creating the recognizer as used for options
recognizer = GestureRecognizer.create_from_options(options)



def norm_land_dist(lm1, lm2):
    """Compute normalized distance between two landmarks."""
    return dist((lm1.x, lm1.y, lm1.z), (lm2.x, lm2.y, lm2.z))



def get_gesture(h, w, result=None, frame=None):
    """Draw landmarks/labels on frame using provided result (or latest)."""
    global _latest_result
    hands = []

    r = result if result is not None else _latest_result
    if r and r.hand_landmarks:

        # iterate hands
        for i, hand in enumerate(r.hand_landmarks):

            hand_data = {
                'relative_landmarks': list((lm.x, lm.y, lm.z) for lm in hand),
                'gesture': None,
            }

            if r.gestures and len(r.gestures) > i and r.gestures[i]:
                gesture = r.gestures[i][0].category_name

                hand_data['gesture'] = gesture
                try:
                    # Additional custom gesture: Pinching
                    if gesture == 'None' and norm_land_dist(hand[4], hand[8]) < 0.08:
                        hand_data['gesture'] = 'Pinching'
                except:
                    pass



                    

            if frame is not None:

                # draw each landmark on that hand
                for x, y, z in hand_data['relative_landmarks']:
                    px = int(x * w)
                    py = int(y * h)
                    cv2.circle(frame, (px,py), 4, (0,255,0), -1)

                    # use first landmark as anchor text point
                    if hand_data['gesture']:

                        first = hand[0]
                        cv2.putText(frame, hand_data['gesture'], (int(first.x*w), int(first.y*h)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            hands.append(hand_data)

        return hands
                    

        # Top gesture: 
        # Category(index=-1,
        #  score=0.6641677021980286,
        #  display_name='',
        #  category_name='Closed_Fist')
        # Hand landmarks:
        #  NormalizedLandmark(x=0.3279569745063782,
        #              y=0.1346130073070526,
        #              z=-0.08733733743429184,
        #              visibility=0.0,
        #              presence=0.0),




def get_gesture_helper(frame):
    global _busy
    # Provide simple backpressure so we don't overload recognize_async
    with _busy_lock:
        if _busy:
            return
        _busy = True

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    recognizer.recognize_async(image, frame_timestamp_ms)

   # Option to run
if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)

        get_gesture_helper(frame)
        

        # Retrieve latest processed output and display from main thread
        display_frame = None
        display_result = None
        with _latest_lock:
            if _latest_frame is not None:
                display_frame = _latest_frame
                display_result = _latest_result
                # Work on a copy to avoid race when callback updates
                display_frame = display_frame.copy()

        if display_frame is not None and display_result is not None:
            h, w = display_frame.shape[:2]
            get_gesture(h, w, None, display_frame)
            cv2.imshow('frame', display_frame)
        else:
            # Show the live frame to avoid a grey window while waiting for first callback
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()