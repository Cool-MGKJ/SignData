"""
Standalone ASL Avatar Dataset Creation Tool

Captures full-body pose, hand landmarks, and facial expressions
for ASL gesture animation dataset generation.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time


@dataclass
class BodyJoint:
    """Represents a single body joint with 3D position."""
    x: float
    y: float
    z: float


@dataclass
class BodyPose:
    """Full body pose with joints and hierarchy."""
    joints: List[List[float]]  # [[x,y,z], ...]
    hierarchy: List[str]  # Joint names in parent-child order


@dataclass
class HandLandmarks:
    """21 landmarks for a single hand."""
    landmarks: List[List[float]]  # [[x,y,z], ...] length = 21


@dataclass
class FaceExpression:
    """Facial expression blendshape weights."""
    blendshapes: Dict[str, float]  # {"smile": 0.6, "browRaise": 0.2, ...}


@dataclass
class FrameData:
    """Single frame of captured data."""
    timestamp: float
    body_pose: BodyPose
    left_hand: Optional[HandLandmarks]
    right_hand: Optional[HandLandmarks]
    face: FaceExpression


@dataclass
class GestureSequence:
    """Complete gesture sequence with label."""
    label: str
    fps: int
    num_frames: int
    frames: List[FrameData]


class ASLAvatarCapture:
    """Main capture class for ASL avatar dataset creation."""
    
    # Body joint hierarchy (MediaPipe Pose landmarks) - Upper body only
    BODY_JOINT_NAMES = [
        "pelvis",           # 23 (left_hip) and 24 (right_hip) midpoint (torso base)
        "spine",            # 23-24 midpoint to 11-12 midpoint
        "neck",             # 11 (left_shoulder) and 12 (right_shoulder) midpoint
        "head",             # 0 (nose)
        "left_shoulder",    # 11
        "left_elbow",      # 13
        "left_wrist",      # 15
        "right_shoulder",  # 12
        "right_elbow",     # 14
        "right_wrist",     # 16
    ]
    
    # MediaPipe Pose landmark indices
    POSE_LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    
    # Facial blendshape names (MediaPipe Face Mesh)
    FACIAL_BLENDSHAPES = [
        "smile",
        "mouthOpen",
        "browRaise",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "jawOpen",
        "mouthSmileLeft",
        "mouthSmileRight",
        "browInnerUp",
        "eyeSquintLeft",
        "eyeSquintRight",
    ]
    
    def __init__(self, camera_index: int = 0, fps: int = 30):
        """
        Initialize capture system.
        
        Args:
            camera_index: Camera device index
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.fps = fps
        self.cap = None
        
        # MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Capture state
        self.is_capturing = False
        self.captured_frames: List[FrameData] = []
        self.start_time: Optional[float] = None
        
    def start_capture(self):
        """Start capturing frames."""
        self.is_capturing = True
        self.captured_frames = []
        self.start_time = time.time()
        
    def stop_capture(self):
        """Stop capturing frames."""
        self.is_capturing = False
        
    def _calculate_pelvis(self, landmarks) -> Tuple[float, float, float]:
        """Calculate pelvis position as midpoint of left and right hips."""
        left_hip = landmarks.landmark[self.POSE_LANDMARKS["left_hip"]]
        right_hip = landmarks.landmark[self.POSE_LANDMARKS["right_hip"]]
        return (
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2,
            (left_hip.z + right_hip.z) / 2
        )
    
    def _calculate_spine(self, landmarks) -> Tuple[float, float, float]:
        """Calculate spine position as midpoint between pelvis and shoulders."""
        pelvis = self._calculate_pelvis(landmarks)
        left_shoulder = landmarks.landmark[self.POSE_LANDMARKS["left_shoulder"]]
        right_shoulder = landmarks.landmark[self.POSE_LANDMARKS["right_shoulder"]]
        shoulder_mid = (
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2
        )
        return (
            (pelvis[0] + shoulder_mid[0]) / 2,
            (pelvis[1] + shoulder_mid[1]) / 2,
            (pelvis[2] + shoulder_mid[2]) / 2
        )
    
    def _calculate_neck(self, landmarks) -> Tuple[float, float, float]:
        """Calculate neck position as midpoint of shoulders."""
        left_shoulder = landmarks.landmark[self.POSE_LANDMARKS["left_shoulder"]]
        right_shoulder = landmarks.landmark[self.POSE_LANDMARKS["right_shoulder"]]
        return (
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2,
            (left_shoulder.z + right_shoulder.z) / 2
        )
    
    def _extract_body_pose(self, pose_landmarks) -> BodyPose:
        """Extract body pose from MediaPipe pose landmarks."""
        if not pose_landmarks:
            # Return empty pose
            return BodyPose(joints=[], hierarchy=self.BODY_JOINT_NAMES)
        
        joints = []
        
        # Calculate derived joints
        pelvis = self._calculate_pelvis(pose_landmarks)
        spine = self._calculate_spine(pose_landmarks)
        neck = self._calculate_neck(pose_landmarks)
        
        # Build joint list in hierarchy order
        joints.append(list(pelvis))  # pelvis
        joints.append(list(spine))   # spine
        joints.append(list(neck))    # neck
        joints.append([pose_landmarks.landmark[self.POSE_LANDMARKS["nose"]].x,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["nose"]].y,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["nose"]].z])  # head
        
        # Left arm
        joints.append([pose_landmarks.landmark[self.POSE_LANDMARKS["left_shoulder"]].x,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["left_shoulder"]].y,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["left_shoulder"]].z])
        joints.append([pose_landmarks.landmark[self.POSE_LANDMARKS["left_elbow"]].x,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["left_elbow"]].y,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["left_elbow"]].z])
        joints.append([pose_landmarks.landmark[self.POSE_LANDMARKS["left_wrist"]].x,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["left_wrist"]].y,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["left_wrist"]].z])
        
        # Right arm
        joints.append([pose_landmarks.landmark[self.POSE_LANDMARKS["right_shoulder"]].x,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["right_shoulder"]].y,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["right_shoulder"]].z])
        joints.append([pose_landmarks.landmark[self.POSE_LANDMARKS["right_elbow"]].x,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["right_elbow"]].y,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["right_elbow"]].z])
        joints.append([pose_landmarks.landmark[self.POSE_LANDMARKS["right_wrist"]].x,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["right_wrist"]].y,
                      pose_landmarks.landmark[self.POSE_LANDMARKS["right_wrist"]].z])
        
        # Note: Legs and lower body joints are excluded - upper body only
        
        return BodyPose(joints=joints, hierarchy=self.BODY_JOINT_NAMES)
    
    def _extract_hand_landmarks(self, hand_landmarks) -> List[List[float]]:
        """Extract 21 landmarks from MediaPipe hand landmarks."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def _extract_face_expression(self, face_landmarks) -> FaceExpression:
        """Extract facial expression blendshapes from face landmarks."""
        # MediaPipe Face Mesh doesn't provide direct blendshapes,
        # so we compute approximations from landmark positions
        
        # Initialize all blendshapes to 0.0
        blendshapes = {name: 0.0 for name in self.FACIAL_BLENDSHAPES}
        
        if not face_landmarks or len(face_landmarks.landmark) < 468:
            return FaceExpression(blendshapes=blendshapes)
        
        landmarks = face_landmarks.landmark
        
        # MediaPipe Face Mesh landmark indices (468 total)
        # Key landmarks for expression detection
        try:
            # Mouth landmarks
            mouth_left = landmarks[61]   # Left mouth corner
            mouth_right = landmarks[291]  # Right mouth corner
            upper_lip_top = landmarks[13]  # Upper lip top
            lower_lip_bottom = landmarks[14]  # Lower lip bottom
            upper_lip_center = landmarks[12]  # Upper lip center
            lower_lip_center = landmarks[15]  # Lower lip center
            
            # Smile: mouth corner separation and upward movement
            mouth_width = abs(mouth_left.x - mouth_right.x)
            mouth_center_y = (upper_lip_center.y + lower_lip_center.y) / 2
            mouth_corner_avg_y = (mouth_left.y + mouth_right.y) / 2
            smile_upward = max(0.0, mouth_center_y - mouth_corner_avg_y)
            blendshapes["smile"] = min(1.0, max(0.0, (mouth_width - 0.03) * 8 + smile_upward * 15))
            
            # Mouth open: vertical distance between lips
            mouth_open = abs(upper_lip_center.y - lower_lip_center.y)
            blendshapes["mouthOpen"] = min(1.0, max(0.0, (mouth_open - 0.008) * 25))
            
            # Asymmetric smile
            left_smile = max(0.0, (mouth_left.y - upper_lip_center.y) * 20)
            right_smile = max(0.0, (mouth_right.y - upper_lip_center.y) * 20)
            blendshapes["mouthSmileLeft"] = min(1.0, left_smile)
            blendshapes["mouthSmileRight"] = min(1.0, right_smile)
        except (IndexError, AttributeError):
            pass
        
        try:
            # Eye landmarks
            # Left eye
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            left_eye_left = landmarks[33]
            left_eye_right = landmarks[133]
            left_eye_center = landmarks[468 // 2]  # Approximate center
            
            # Right eye
            right_eye_top = landmarks[386]
            right_eye_bottom = landmarks[374]
            right_eye_left = landmarks[362]
            right_eye_right = landmarks[263]
            
            # Eye blink: Eye Aspect Ratio (EAR)
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            left_eye_width = abs(left_eye_left.x - left_eye_right.x)
            left_ear = left_eye_height / (left_eye_width + 1e-6)
            # Normal eye EAR ~0.25-0.3, closed ~0.1
            blendshapes["eyeBlinkLeft"] = min(1.0, max(0.0, (0.25 - left_ear) * 6))
            
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
            right_eye_width = abs(right_eye_left.x - right_eye_right.x)
            right_ear = right_eye_height / (right_eye_width + 1e-6)
            blendshapes["eyeBlinkRight"] = min(1.0, max(0.0, (0.25 - right_ear) * 6))
            
            # Eye squint (partial closure)
            blendshapes["eyeSquintLeft"] = min(1.0, blendshapes["eyeBlinkLeft"] * 0.6)
            blendshapes["eyeSquintRight"] = min(1.0, blendshapes["eyeBlinkRight"] * 0.6)
        except (IndexError, AttributeError):
            pass
        
        try:
            # Brow landmarks
            left_brow_outer = landmarks[107]
            left_brow_inner = landmarks[107]  # Approximate
            left_eye_top = landmarks[159]
            right_brow_outer = landmarks[336]
            right_eye_top = landmarks[386]
            
            # Brow raise: distance between brow and eye
            left_brow_raise = max(0.0, (left_eye_top.y - left_brow_outer.y) - 0.015) * 20
            right_brow_raise = max(0.0, (right_eye_top.y - right_brow_outer.y) - 0.015) * 20
            blendshapes["browRaise"] = min(1.0, (left_brow_raise + right_brow_raise) / 2)
            blendshapes["browInnerUp"] = min(1.0, blendshapes["browRaise"] * 0.8)
        except (IndexError, AttributeError):
            pass
        
        try:
            # Jaw open: chin to lower lip distance
            chin = landmarks[175]  # Chin point
            lower_lip = landmarks[14]
            jaw_open = abs(chin.y - lower_lip.y)
            blendshapes["jawOpen"] = min(1.0, max(0.0, (jaw_open - 0.01) * 20))
        except (IndexError, AttributeError):
            pass
        
        # Ensure all values are normalized
        for key in blendshapes:
            blendshapes[key] = max(0.0, min(1.0, blendshapes[key]))
        
        return FaceExpression(blendshapes=blendshapes)
    
    def process_frame(self, frame) -> Optional[FrameData]:
        """
        Process a single frame and extract all data.
        
        Returns:
            FrameData if successful, None otherwise
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        # Extract body pose
        body_pose = self._extract_body_pose(pose_results.pose_landmarks)
        
        # Extract hand landmarks
        left_hand = None
        right_hand = None
        
        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                hand_type = hands_results.multi_handedness[idx].classification[0].label
                landmarks = self._extract_hand_landmarks(hand_landmarks)
                
                if hand_type == "Left":
                    left_hand = HandLandmarks(landmarks=landmarks)
                elif hand_type == "Right":
                    right_hand = HandLandmarks(landmarks=landmarks)
        
        # Extract face expression
        face_expression = self._extract_face_expression(face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None)
        
        # Calculate timestamp
        timestamp = time.time() - self.start_time if self.start_time else 0.0
        
        return FrameData(
            timestamp=timestamp,
            body_pose=body_pose,
            left_hand=left_hand,
            right_hand=right_hand,
            face=face_expression
        )
    
    def capture_frame(self, frame):
        """Capture a frame if capturing is active."""
        if self.is_capturing:
            frame_data = self.process_frame(frame)
            if frame_data:
                self.captured_frames.append(frame_data)
    
    def get_sequence(self, label: str) -> GestureSequence:
        """
        Get the captured sequence as a GestureSequence.
        
        Args:
            label: Gesture label (lowercase, snake_case)
        
        Returns:
            GestureSequence object
        """
        return GestureSequence(
            label=label.lower().replace(" ", "_"),
            fps=self.fps,
            num_frames=len(self.captured_frames),
            frames=self.captured_frames
        )
    
    def clear_capture(self):
        """Clear captured frames."""
        self.captured_frames = []
        self.start_time = None
    
    def release(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()
