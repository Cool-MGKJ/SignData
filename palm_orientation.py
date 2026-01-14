"""
Palm Orientation Tracking Module

This module calculates the palm's orientation (Pitch, Yaw, Roll) in degrees
from hand landmarks using the palm normal vector.

The palm plane is defined by:
- Landmark 0 (Wrist)
- Landmark 5 (Index MCP)
- Landmark 17 (Pinky MCP)

The normal vector is calculated as the cross product of vectors from wrist
to index MCP and wrist to pinky MCP.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def get_palm_orientation(landmarks: List[Tuple[float, float, float]]) -> Optional[Dict[str, float]]:
    """
    Calculate palm orientation (Pitch, Yaw, Roll) in degrees from hand landmarks.
    
    Uses landmarks 0 (Wrist), 5 (Index MCP), and 17 (Pinky MCP) to define the palm plane.
    Calculates the normal vector and converts it to Euler angles.
    
    Args:
        landmarks: List of 21 (x, y, z) tuples representing hand landmarks
    
    Returns:
        Dictionary with keys 'pitch', 'yaw', 'roll' (in degrees), or None if invalid
    """
    if landmarks is None or len(landmarks) < 18:
        return None
    
    # Extract key landmarks
    wrist = np.array(landmarks[0])      # Landmark 0: Wrist
    index_mcp = np.array(landmarks[5])   # Landmark 5: Index MCP
    pinky_mcp = np.array(landmarks[17]) # Landmark 17: Pinky MCP
    
    # Calculate vectors from wrist to index MCP and wrist to pinky MCP
    vec_0_to_5 = index_mcp - wrist   # Vector from wrist to index MCP
    vec_0_to_17 = pinky_mcp - wrist  # Vector from wrist to pinky MCP
    
    # Calculate normal vector using cross product
    # Normal = vec_0_to_5 × vec_0_to_17
    normal = np.cross(vec_0_to_5, vec_0_to_17)
    
    # Normalize the normal vector
    norm_magnitude = np.linalg.norm(normal)
    if norm_magnitude < 1e-6:
        # Vectors are parallel or zero, cannot determine orientation
        return None
    
    normal_normalized = normal / norm_magnitude
    
    # Convert normal vector to Euler angles (Pitch, Yaw, Roll)
    pitch, yaw, roll = normal_to_euler_angles(normal_normalized)
    
    return {
        'pitch': pitch,
        'yaw': yaw,
        'roll': roll
    }


def normal_to_euler_angles(normal: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a normalized 3D normal vector to Euler angles (Pitch, Yaw, Roll).
    
    Euler angle convention:
    - Pitch: Rotation around X-axis (tilt up/down) - range: [-90, 90] degrees
    - Yaw: Rotation around Y-axis (turn left/right) - range: [-180, 180] degrees
    - Roll: Rotation around Z-axis (tilt side-to-side) - range: [-180, 180] degrees
    
    The normal vector represents the direction perpendicular to the palm plane.
    We convert this to angles relative to the standard coordinate system where:
    - X-axis: right
    - Y-axis: down
    - Z-axis: forward (toward camera)
    
    Args:
        normal: Normalized 3D vector (numpy array of shape (3,))
    
    Returns:
        Tuple of (pitch, yaw, roll) in degrees
    """
    nx, ny, nz = normal[0], normal[1], normal[2]
    
    # Calculate pitch (rotation around X-axis)
    # Pitch is the angle between the normal's projection on YZ plane and Z-axis
    pitch_rad = np.arcsin(np.clip(-ny, -1.0, 1.0))  # Negative Y because Y points down
    pitch = np.degrees(pitch_rad)
    
    # Calculate yaw (rotation around Y-axis)
    # Yaw is the angle between the normal's projection on XZ plane and Z-axis
    if abs(nz) < 1e-6:
        # Normal is perpendicular to Z-axis, yaw is undefined (use 0)
        yaw = 0.0
    else:
        yaw_rad = np.arctan2(nx, nz)
        yaw = np.degrees(yaw_rad)
    
    # Calculate roll (rotation around Z-axis)
    # Roll is the angle of the normal's projection on XY plane
    # We use the angle between the normal's XY projection and X-axis
    xy_magnitude = np.sqrt(nx * nx + ny * ny)
    if xy_magnitude < 1e-6:
        # Normal is parallel to Z-axis, roll is undefined (use 0)
        roll = 0.0
    else:
        roll_rad = np.arctan2(ny, nx)
        roll = np.degrees(roll_rad)
    
    return (pitch, yaw, roll)


def calculate_angle_delta(
    current_angles: Dict[str, float],
    last_angles: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate the absolute difference between current and last angles.
    
    Args:
        current_angles: Dictionary with 'pitch', 'yaw', 'roll' keys
        last_angles: Dictionary with 'pitch', 'yaw', 'roll' keys
    
    Returns:
        Dictionary with absolute differences for each angle
    """
    return {
        'pitch': abs(current_angles['pitch'] - last_angles['pitch']),
        'yaw': abs(current_angles['yaw'] - last_angles['yaw']),
        'roll': abs(current_angles['roll'] - last_angles['roll'])
    }


def has_significant_change(
    current_angles: Dict[str, float],
    last_angles: Dict[str, float],
    threshold_degrees: float = 5.0
) -> bool:
    """
    Check if any angle has changed by at least the threshold.
    
    Args:
        current_angles: Current palm angles dictionary
        last_angles: Last saved palm angles dictionary
        threshold_degrees: Minimum change required (default: 5.0)
    
    Returns:
        True if any angle changed by >= threshold_degrees, False otherwise
    """
    if last_angles is None:
        # First sample, always return True
        return True
    
    deltas = calculate_angle_delta(current_angles, last_angles)
    
    # Check if any angle changed by at least threshold
    return (deltas['pitch'] >= threshold_degrees or
            deltas['yaw'] >= threshold_degrees or
            deltas['roll'] >= threshold_degrees)





