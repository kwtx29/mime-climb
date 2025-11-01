# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
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
                for x, y in hand_data['relative_landmarks']:
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

    result = recognizer.recognize(image)
    
    h, w, _ = frame.shape

    get_gesture(result, h, w, frame)



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

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()