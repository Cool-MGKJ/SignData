"""
Webcam capture and MediaPipe hand landmark detection module.

This module handles:
- Webcam initialization and frame capture
- MediaPipe Hands initialization and processing
- Hand landmark extraction (3D coordinates)
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple


class HandCapture:
    """Manages webcam capture and MediaPipe hand detection."""
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize the hand capture system.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def initialize(self) -> bool:
        """
        Initialize the camera and MediaPipe Hands.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Initialize MediaPipe Hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,  # Detect up to 2 hands
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
            
            return True
        except Exception as e:
            print(f"Error initializing capture: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the webcam.
        
        Returns:
            BGR frame as numpy array, or None if failed
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Process a frame to detect hand landmarks.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            Tuple of (annotated_frame, landmarks_list)
            landmarks_list contains dicts with keys:
                - 'hand': 'left' or 'right'
                - 'landmarks': list of (x, y, z) tuples (21 points per hand)
        """
        if self.hands is None:
            return frame, []
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for drawing
        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        landmarks_list = []
        
        # Extract landmarks for each detected hand
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Determine which hand (left/right from camera's perspective)
                hand_label = handedness.classification[0].label
                
                # Extract 3D coordinates
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    # MediaPipe provides normalized coordinates (0-1) and z depth
                    landmarks_3d.append((
                        landmark.x,  # Normalized x (0-1)
                        landmark.y,  # Normalized y (0-1)
                        landmark.z   # Depth (relative, can be negative)
                    ))
                
                landmarks_list.append({
                    'hand': hand_label.lower(),  # 'Left' or 'Right'
                    'landmarks': landmarks_3d
                })
                
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_frame, landmarks_list
    
    def get_landmarks(self, frame: np.ndarray) -> List[dict]:
        """
        Extract hand landmarks from a frame without drawing.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            List of dicts with 'hand' and 'landmarks' keys
        """
        if self.hands is None:
            return []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process
        results = self.hands.process(rgb_frame)
        
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    landmarks_3d.append((landmark.x, landmark.y, landmark.z))
                
                landmarks_list.append({
                    'hand': hand_label.lower(),
                    'landmarks': landmarks_3d
                })
        
        return landmarks_list
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
        if self.hands is not None:
            self.hands.close()

