import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import threading
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

class SmoothHandTracker:
    def __init__(self):
        # Initialize MediaPipe
        self.cap = cv2.VideoCapture(0)
        self.hands = mphands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Create tkinter window
        self.root = tk.Tk()
        self.root.title("Smooth Hand Tracker")
        
        # Make window transparent and always on top
        self.root.attributes('-alpha', 0.9)
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Set window to fullscreen
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        self.root.configure(bg='black')
        
        # Make black transparent
        self.root.attributes('-transparentcolor', 'black')
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Bind keys
        self.root.bind('<Escape>', self.close_app)
        self.root.bind('<KeyPress-q>', self.close_app)
        self.root.focus_set()
        
        self.running = True
        self.current_drawings = []
        
    def close_app(self, event=None):
        self.running = False
        self.root.quit()
        
    def draw_hand_landmarks(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Process frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Clear previous drawings
            for item in self.current_drawings:
                try:
                    self.canvas.delete(item)
                except:
                    pass
            self.current_drawings.clear()
            
            # Draw new landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw connections first (behind landmarks)
                    connections = mphands.HAND_CONNECTIONS
                    for connection in connections:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        start_point = hand_landmarks.landmark[start_idx]
                        end_point = hand_landmarks.landmark[end_idx]
                        
                        start_x = int(start_point.x * self.screen_width)
                        start_y = int(start_point.y * self.screen_height)
                        end_x = int(end_point.x * self.screen_width)
                        end_y = int(end_point.y * self.screen_height)
                        
                        line = self.canvas.create_line(
                            start_x, start_y, end_x, end_y,
                            fill='#00FFFF', width=3, smooth=True,
                            capstyle=tk.ROUND
                        )
                        self.current_drawings.append(line)
                    
                    # Draw landmarks on top
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        x = int(landmark.x * self.screen_width)
                        y = int(landmark.y * self.screen_height)
                        
                        # Different colors for different landmark types
                        if i in [4, 8, 12, 16, 20]:  # Fingertips
                            color = '#FF0000'  # Red
                            size = 6
                        elif i in [0]:  # Wrist
                            color = '#FFFF00'  # Yellow
                            size = 8
                        else:
                            color = '#FF69B4'  # Pink
                            size = 4
                        
                        circle = self.canvas.create_oval(
                            x-size, y-size, x+size, y+size,
                            fill=color, outline='white', width=1
                        )
                        self.current_drawings.append(circle)
            
            # Smooth update
            try:
                self.root.update_idletasks()
            except:
                break
            
            time.sleep(1/30)  # 30 FPS
    
    def run(self):
        # Start tracking thread
        tracking_thread = threading.Thread(target=self.draw_hand_landmarks, daemon=True)
        tracking_thread.start()
        
        # Run GUI
        try:
            self.root.mainloop()
        except:
            pass
        
        # Cleanup
        self.running = False
        self.cap.release()

if __name__ == "__main__":
    tracker = SmoothHandTracker()
    tracker.run()