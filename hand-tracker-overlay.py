#Required things
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands=mp.solutions.hands

cap=cv2.VideoCapture(0)
hands=mphands.Hands()

# Get screen dimensions (you might need to adjust these for your monitor)
screen_width = 1920  # Change this to your screen width
screen_height = 1080  # Change this to your screen height

while True:
    data, image = cap.read()
    if not data:
        break
        
    # Flip the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Process hand detection
    results = hands.process(image)
    
    # Convert back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create a green screen background (for chroma key effect)
    # Use bright green (0, 255, 0) - you can make this "transparent" with OBS or other software
    overlay = np.full((screen_height, screen_width, 3), (0, 255, 0), dtype=np.uint8)
    
    # Draw only hand landmarks on black background (no webcam feed)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks directly on the overlay
            # MediaPipe landmarks are normalized (0-1), so they'll scale to overlay size automatically
            mp_drawing.draw_landmarks(
                overlay,
                hand_landmarks,
                mphands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
    # Create fullscreen window
    cv2.namedWindow('Hand Tracking Overlay', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Hand Tracking Overlay', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Display only the hand overlay (no webcam feed)
    cv2.imshow('Hand Tracking Overlay', overlay)
    
    # Press 'q' or ESC to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 27 is ESC key
        break

# Clean up
cap.release()
cv2.destroyAllWindows()