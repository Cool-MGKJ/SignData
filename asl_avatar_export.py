"""
JSON Export Module for ASL Avatar Dataset

Exports gesture sequences in the specified JSON format.
"""

import json
import os
from typing import Dict, Any
from asl_avatar_capture import GestureSequence, FrameData, BodyPose, HandLandmarks, FaceExpression


def frame_to_dict(frame: FrameData) -> Dict[str, Any]:
    """Convert FrameData to dictionary format."""
    result = {
        "timestamp": round(frame.timestamp, 3),
        "body_pose": {
            "joints": frame.body_pose.joints,
            "hierarchy": frame.body_pose.hierarchy
        }
    }
    
    # Add hand landmarks (always 21 landmarks, zeros if not detected)
    if frame.left_hand and len(frame.left_hand.landmarks) == 21:
        result["left_hand"] = frame.left_hand.landmarks
    else:
        result["left_hand"] = [[0.0, 0.0, 0.0]] * 21
    
    if frame.right_hand and len(frame.right_hand.landmarks) == 21:
        result["right_hand"] = frame.right_hand.landmarks
    else:
        result["right_hand"] = [[0.0, 0.0, 0.0]] * 21
    
    # Add face expression
    result["face"] = {
        "blendshapes": frame.face.blendshapes
    }
    
    return result


def sequence_to_dict(sequence: GestureSequence) -> Dict[str, Any]:
    """Convert GestureSequence to dictionary format for JSON export."""
    return {
        "label": sequence.label,
        "fps": sequence.fps,
        "num_frames": sequence.num_frames,
        "frames": [frame_to_dict(frame) for frame in sequence.frames]
    }


def export_gesture(sequence: GestureSequence, output_dir: str) -> str:
    """
    Export a single gesture sequence to JSON file.
    
    Args:
        sequence: GestureSequence to export
        output_dir: Output directory path
    
    Returns:
        Path to exported file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate label
    label = sequence.label.lower().replace(" ", "_").replace("-", "_")
    if not label or not label.replace("_", "").isalnum():
        raise ValueError(f"Invalid label: {sequence.label}. Must be lowercase alphanumeric with underscores.")
    
    # Create filename
    filename = f"{label}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert to dict and export
    data = sequence_to_dict(sequence)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def validate_sequence(sequence: GestureSequence):
    """
    Validate a gesture sequence before export.
    
    Returns:
        (is_valid, error_message)
    """
    if not sequence.label:
        return False, "Label is required"
    
    if sequence.num_frames == 0:
        return False, "Sequence has no frames"
    
    # Check label format
    label_clean = sequence.label.lower().replace("_", "").replace("-", "")
    if not label_clean.isalnum():
        return False, f"Invalid label format: {sequence.label}. Must be lowercase alphanumeric with underscores."
    
    # Check for minimum data
    frames_with_pose = sum(1 for f in sequence.frames if f.body_pose.joints)
    if frames_with_pose == 0:
        return False, "No frames with valid body pose"
    
    return True, ""
