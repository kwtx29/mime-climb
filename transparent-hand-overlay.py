import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

class TransparentHandTracker:
    def __init__(self):
        # Initialize MediaPipe
        self.cap = cv2.VideoCapture(0)
        self.hands = mphands.Hands()
        
        # Create tkinter window
        self.root = tk.Tk()
        self.root.title("Hand Tracker Overlay")
        
        # Make window transparent and always on top
        self.root.attributes('-alpha', 0.8)  # 80% opacity
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        # Set window to fullscreen
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        
        # Make window click-through (optional)
        self.root.attributes('-transparentcolor', 'black')
        
        # Create canvas for drawing
        self.canvas = tk.Canvas(
            self.root, 
            width=self.screen_width, 
            height=self.screen_height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Bind escape key to close
        self.root.bind('<Escape>', self.close_app)
        self.root.bind('<KeyPress-q>', self.close_app)
        self.root.focus_set()
        
        self.running = True
        
    def close_app(self, event=None):
        self.running = False
        self.root.quit()
        
    def draw_hand_landmarks(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Flip and process frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Only clear and redraw if hands are detected or if we need to clear previous drawings
            self.canvas.delete("all")
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                # Batch all drawing operations
                drawings = []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw connections
                    connections = mphands.HAND_CONNECTIONS
                    for connection in connections:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        start_point = hand_landmarks.landmark[start_idx]
                        end_point = hand_landmarks.landmark[end_idx]
                        
                        # Convert to screen coordinates
                        start_x = int(start_point.x * self.screen_width)
                        start_y = int(start_point.y * self.screen_height)
                        end_x = int(end_point.x * self.screen_width)
                        end_y = int(end_point.y * self.screen_height)
                        
                        # Draw line with smoother appearance
                        self.canvas.create_line(
                            start_x, start_y, end_x, end_y,
                            fill='cyan', width=2, smooth=True
                        )
                    
                    # Draw landmarks
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * self.screen_width)
                        y = int(landmark.y * self.screen_height)
                        
                        # Draw point with smoother edges
                        self.canvas.create_oval(
                            x-4, y-4, x+4, y+4,
                            fill='red', outline='white', width=1
                        )
            
            # Update display less frequently to reduce flicker
            self.root.update_idletasks()
            time.sleep(0.05)  # Slower refresh rate to reduce flicker
    
    def run(self):
        # Start hand tracking in separate thread
        tracking_thread = threading.Thread(target=self.draw_hand_landmarks)
        tracking_thread.daemon = True
        tracking_thread.start()
        
        # Start GUI
        self.root.mainloop()
        
        # Cleanup
        self.cap.release()

if __name__ == "__main__":
    tracker = TransparentHandTracker()
    tracker.run()