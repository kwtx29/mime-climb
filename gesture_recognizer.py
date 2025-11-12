# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import threading



BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Thread-safe storage for latest callback output
_latest_lock = threading.Lock()
_latest_pair = None  # tuple[np.ndarray (BGR frame), GestureRecognizerResult]

# Create a gesture recognizer instance with the live stream mode:

def get_result(result, output_image: mp.Image, timestamp_ms: int):
    """Result listener for LIVE_STREAM; stores frame/result for main thread to draw."""
    global _latest_pair
    # Convert MediaPipe Image (RGB) to numpy BGR for OpenCV drawing
    try:
        frame_rgb = output_image.numpy_view()
    except AttributeError:
        # Fallback for older MP versions
        frame_rgb = output_image.numpy()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Store for main thread display
    with _latest_lock:
        _latest_pair = (frame_bgr, result)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result)



# IMPORTANT: Use the same API family for creating the recognizer as used for options
recognizer = GestureRecognizer.create_from_options(options)







def get_gesture(h, w, result = None,  frame=None):
    global _latest_pair
    hands = []
    if result:
        pass
    else:
        result = _latest_pair[1] if _latest_pair else None
    if result and result.hand_landmarks:

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
            if _latest_pair is not None:
                display_frame, display_result = _latest_pair
                # Work on a copy to avoid race when callback updates
                display_frame = display_frame.copy()

        if display_frame is not None and display_result is not None:
            h, w = display_frame.shape[:2]
            get_gesture(h, w, display_result, display_frame)
            cv2.imshow('frame', display_frame)
        else:
            # Show the live frame to avoid a grey window while waiting for first callback
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()