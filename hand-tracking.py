import cv2
import time
import numpy as np
import mediapipe as mp

# hand-tracking.py
# Requires: mediapipe, opencv-python, numpy
# Install: pip install mediapipe opencv-python numpy


# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Utility: convert normalized landmark to pixel coords
def to_pixel_landmark(landmark, image_shape):
    h, w = image_shape[:2]
    return int(landmark.x * w), int(landmark.y * h)

# Utility: angle at point b between points a-b-c (in degrees)
def angle_between_points(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    prev_time = 0.0
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Draw pose landmarks (arms/shoulders)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2),
                )

                lm = results.pose_landmarks.landmark
                img_h, img_w = frame.shape[:2]

                # Left arm indexes in Pose: shoulder=11, elbow=13, wrist=15
                try:
                    left_sh = to_pixel_landmark(lm[11], frame.shape)
                    left_el = to_pixel_landmark(lm[13], frame.shape)
                    left_wr = to_pixel_landmark(lm[15], frame.shape)
                    left_elbow_angle = angle_between_points(left_sh, left_el, left_wr)
                    cv2.putText(frame, f"L_elbow: {int(left_elbow_angle)}",
                                (left_el[0] - 40, left_el[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                except:
                    pass

                # Right arm indexes in Pose: shoulder=12, elbow=14, wrist=16
                try:
                    right_sh = to_pixel_landmark(lm[12], frame.shape)
                    right_el = to_pixel_landmark(lm[14], frame.shape)
                    right_wr = to_pixel_landmark(lm[16], frame.shape)
                    right_elbow_angle = angle_between_points(right_sh, right_el, right_wr)
                    cv2.putText(frame, f"R_elbow: {int(right_elbow_angle)}",
                                (right_el[0] - 40, right_el[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                except:
                    pass

            # Draw hands
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, "Left hand", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (121,22,76), 2)

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),
                )
                cv2.putText(frame, "Right hand", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245,117,66), 2)

            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("Hand & Arm Tracker (press ESC to quit)", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()