"""
JSON Gesture File Loader

Loads ASL gesture sequences from JSON files created by the capture tool.
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BodyPose:
    """Body pose with joints and hierarchy."""
    joints: List[List[float]]  # [[x,y,z], ...]
    hierarchy: List[str]


@dataclass
class FaceExpression:
    """Facial expression blendshapes."""
    blendshapes: Dict[str, float]


@dataclass
class FrameData:
    """Single frame of gesture data."""
    timestamp: float
    body_pose: BodyPose
    left_hand: List[List[float]]  # 21 landmarks
    right_hand: List[List[float]]  # 21 landmarks
    face: FaceExpression


@dataclass
class GestureSequence:
    """Complete gesture sequence loaded from JSON."""
    label: str
    fps: int
    num_frames: int
    frames: List[FrameData]
    duration: float  # Total duration in seconds


class GestureLoader:
    """Loads gesture sequences from JSON files."""
    
    def __init__(self, dataset_dir: str = "asl_avatar_dataset"):
        """
        Initialize gesture loader.
        
        Args:
            dataset_dir: Directory containing JSON gesture files
        """
        self.dataset_dir = dataset_dir
        self._cache: Dict[str, GestureSequence] = {}
    
    def load_gesture(self, label: str) -> Optional[GestureSequence]:
        """
        Load a gesture sequence by label.
        
        Args:
            label: Gesture label (e.g., "hello", "sorry")
        
        Returns:
            GestureSequence if found, None otherwise
        """
        # Check cache first
        if label in self._cache:
            return self._cache[label]
        
        # Normalize label
        label = label.lower().replace(" ", "_").replace("-", "_")
        
        # Construct file path
        filename = f"{label}.json"
        filepath = os.path.join(self.dataset_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse frames
            frames = []
            for frame_data in data.get("frames", []):
                # Parse body pose
                body_pose_data = frame_data.get("body_pose", {})
                body_pose = BodyPose(
                    joints=body_pose_data.get("joints", []),
                    hierarchy=body_pose_data.get("hierarchy", [])
                )
                
                # Parse hands (ensure 21 landmarks each)
                left_hand = frame_data.get("left_hand", [])
                right_hand = frame_data.get("right_hand", [])
                
                # Pad with zeros if needed
                if len(left_hand) < 21:
                    left_hand.extend([[0.0, 0.0, 0.0]] * (21 - len(left_hand)))
                if len(right_hand) < 21:
                    right_hand.extend([[0.0, 0.0, 0.0]] * (21 - len(right_hand)))
                
                # Parse face expression
                face_data = frame_data.get("face", {})
                face_expression = FaceExpression(
                    blendshapes=face_data.get("blendshapes", {})
                )
                
                frame = FrameData(
                    timestamp=frame_data.get("timestamp", 0.0),
                    body_pose=body_pose,
                    left_hand=left_hand[:21],  # Ensure exactly 21
                    right_hand=right_hand[:21],  # Ensure exactly 21
                    face=face_expression
                )
                frames.append(frame)
            
            # Calculate duration
            if frames:
                duration = frames[-1].timestamp - frames[0].timestamp
                if duration <= 0:
                    duration = len(frames) / data.get("fps", 30)
            else:
                duration = 0.0
            
            sequence = GestureSequence(
                label=data.get("label", label),
                fps=data.get("fps", 30),
                num_frames=data.get("num_frames", len(frames)),
                frames=frames,
                duration=duration
            )
            
            # Cache it
            self._cache[label] = sequence
            
            return sequence
            
        except Exception as e:
            print(f"Error loading gesture '{label}': {e}")
            return None
    
    def list_available_gestures(self) -> List[str]:
        """
        List all available gesture labels in the dataset.
        
        Returns:
            List of gesture labels (without .json extension)
        """
        if not os.path.exists(self.dataset_dir):
            return []
        
        gestures = []
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith(".json"):
                label = filename[:-5]  # Remove .json
                gestures.append(label)
        
        return sorted(gestures)
    
    def clear_cache(self):
        """Clear the gesture cache."""
        self._cache.clear()
