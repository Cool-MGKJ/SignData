"""
Hand landmark normalization module.

This module provides functions to normalize MediaPipe hand landmarks
into a consistent 3D space with configurable spacing and point count.
"""

import numpy as np
from typing import List, Tuple, Optional

# Optional sklearn dependency for standardization
try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    StandardScaler = None


def normalize_landmarks_to_3d_space(
    landmarks: List[Tuple[float, float, float]],
    num_points: int,
    spacing_ratio: float = 1.0,
    preserve_depth: bool = True
) -> List[Tuple[float, float, float]]:
    """
    Normalize hand landmarks into a consistent 3D space while preserving MediaPipe relative depth.
    
    This function takes raw MediaPipe landmarks (which vary based on
    person size and distance to camera) and normalizes them to a
    consistent coordinate space.
    
    Normalization steps (when preserve_depth=True, recommended for MediaPipe):
    1. Wrist centering: Subtract wrist (index 0) from all points
    2. 2D palm scaling: Scale by 2D (x,y) distance from wrist to middle finger base (index 9)
       This preserves MediaPipe's relative z-depth structure
    3. Apply spacing_ratio for overall scaling
    4. Ensure exactly num_points are returned
    
    Normalization steps (when preserve_depth=False, legacy method):
    1. Translate to origin (subtract centroid)
    2. Scale to unit size (normalize by maximum distance from origin)
    3. Apply spacing_ratio for overall scaling
    4. Ensure exactly num_points are returned
    
    Args:
        landmarks: List of (x, y, z) tuples from MediaPipe
        num_points: Target number of points to return (should match input length)
        spacing_ratio: Scaling factor for the normalized space (default: 1.0)
                      Values > 1.0 make the space larger, < 1.0 make it smaller
        preserve_depth: If True, use wrist-based normalization preserving MediaPipe relative depth (default: True)
                       If False, use centroid-based normalization (legacy, may flatten z-depth)
    
    Returns:
        List of normalized (x, y, z) tuples with exactly num_points elements
    """
    if not landmarks:
        # Return zeros if no landmarks provided
        return [(0.0, 0.0, 0.0)] * num_points
    
    # Convert to numpy array for easier manipulation
    points = np.array(landmarks, dtype=np.float32)
    
    if preserve_depth and len(points) >= 10:  # Need at least 10 points (wrist=0, middle_base=9)
        # MediaPipe-aware normalization that preserves relative z-depth
        # MediaPipe hand landmarks: index 0 = wrist, index 9 = middle finger MCP
        
        # Step 1: Wrist centering (translation invariance)
        wrist = points[0].copy()
        centered_points = points - wrist
        
        # Step 2: Palm scaling using 2D distance (preserves relative z-depth)
        # Use wrist (index 0) to middle finger base (index 9) 2D distance
        if len(centered_points) > 9:
            middle_base_centered = centered_points[9]
            
            # Calculate 2D palm scale (x,y only) - MediaPipe z has different units
            xy_vector = middle_base_centered[:2]  # Only x,y components
            xy_scale = np.linalg.norm(xy_vector)
            
            if xy_scale < 1e-6:
                xy_scale = 1.0
            
            # Scale all coordinates by 2D palm distance (preserves relative z relationships)
            scaled_points = centered_points / xy_scale
        else:
            # Fallback if not enough points
            scaled_points = centered_points
        
    else:
        # Legacy normalization (centroid-based, treats x,y,z equally)
        # Step 1: Translate to origin (center at centroid)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Step 2: Scale to unit size
        # Find the maximum distance from origin to any point
        distances = np.linalg.norm(centered_points, axis=1)
        max_distance = np.max(distances) if len(distances) > 0 else 1.0
        
        # Avoid division by zero
        if max_distance < 1e-6:
            max_distance = 1.0
        
        # Normalize to unit sphere/cube
        scaled_points = centered_points / max_distance
    
    # Step 3: Apply spacing ratio for overall scaling
    scaled_points = scaled_points * spacing_ratio
    
    # Step 4: Ensure we have exactly num_points
    # If we have fewer points, pad with zeros
    # If we have more points, truncate (shouldn't happen with MediaPipe)
    current_count = len(scaled_points)
    
    if current_count < num_points:
        # Pad with zeros
        padding = np.zeros((num_points - current_count, 3), dtype=np.float32)
        scaled_points = np.vstack([scaled_points, padding])
    elif current_count > num_points:
        # Truncate to num_points
        scaled_points = scaled_points[:num_points]
    
    # Convert back to list of tuples
    result = [(float(x), float(y), float(z)) for x, y, z in scaled_points]
    
    return result


def normalize_multiple_hands(
    landmarks_list: List[dict],
    num_points_per_hand: int = 21,
    spacing_ratio: float = 1.0,
    preserve_depth: bool = True
) -> Optional[List[Tuple[float, float, float]]]:
    """
    Normalize landmarks from one or two hands into a single list.
    
    If two hands are detected, concatenate their normalized landmarks.
    If one hand is detected, use only that hand.
    If no hands are detected, return None.
    
    Args:
        landmarks_list: List of dicts with 'hand' and 'landmarks' keys
        num_points_per_hand: Number of landmarks per hand (MediaPipe: 21)
        spacing_ratio: Scaling factor for normalization
        preserve_depth: If True, preserve MediaPipe relative z-depth (default: True)
    
    Returns:
        List of normalized (x, y, z) tuples, or None if no hands detected
    """
    if not landmarks_list:
        return None
    
    # Normalize each hand separately (each hand normalized independently)
    normalized_hands = []
    for hand_data in landmarks_list:
        landmarks = hand_data['landmarks']
        normalized = normalize_landmarks_to_3d_space(
            landmarks,
            num_points_per_hand,
            spacing_ratio,
            preserve_depth=preserve_depth
        )
        normalized_hands.append(normalized)
    
    # Concatenate all hands into a single list
    all_points = []
    for hand_points in normalized_hands:
        all_points.extend(hand_points)
    
    return all_points


def normalize_hand_data(landmarks: List[Tuple[float, float, float]], apply_rotation: bool = True) -> List[Tuple[float, float, float]]:
    """
    Normalize hand landmarks to a "Standard Hand" representation.
    
    This function implements a normalization pipeline that minimizes intra-class spread
    by transforming all hands to a standard size and orientation:
    
    1. Translation (Zero-Centering): Move the hand so that the Wrist (Index 0) is at (0, 0, 0)
    2. Scaling (Standard Size): Scale all coordinates by the Euclidean distance from 
       Wrist (Index 0) to Middle Finger MCP (Index 9)
    3. Orientation Alignment (Optional): Rotate the hand so that the vector from wrist 
       to middle finger base points along the positive Y-axis
    
    Args:
        landmarks: List of 21 (x, y, z) tuples from MediaPipe hand landmarks
        apply_rotation: If True, align rotation so wrist-to-middle-MCP points along +Y axis (default: True)
    
    Returns:
        List of normalized (x, y, z) tuples with wrist at (0, 0, 0) and standard size
    """
    if not landmarks or len(landmarks) < 21:
        # Return zeros if invalid input
        return [(0.0, 0.0, 0.0)] * 21
    
    # Convert to numpy array
    points = np.array(landmarks, dtype=np.float32)
    
    # Step 1: Translation (Zero-Centering)
    # Move wrist (index 0) to origin (0, 0, 0)
    wrist = points[0].copy()
    centered_points = points - wrist
    
    # Step 2: Scaling (Standard Size)
    # Calculate 3D Euclidean distance from Wrist (0) to Middle Finger MCP (9)
    if len(centered_points) > 9:
        middle_mcp = centered_points[9]
        # Use 3D Euclidean distance for scaling (includes x, y, z)
        scale = np.linalg.norm(middle_mcp)
        
        if scale < 1e-6:
            # Avoid division by zero
            scale = 1.0
        
        # Divide all coordinates by this distance
        scaled_points = centered_points / scale
    else:
        # Fallback if not enough points
        scaled_points = centered_points
    
    # Step 3: Orientation Alignment (Optional)
    if apply_rotation and len(scaled_points) > 9:
        # Get the vector from wrist (now at origin) to middle finger MCP
        wrist_to_middle = scaled_points[9]
        
        # Target direction is positive Y-axis: (0, 1, 0)
        target_direction = np.array([0.0, 1.0, 0.0])
        
        # Calculate rotation needed to align wrist-to-middle with +Y axis
        # Using cross product to find rotation axis and angle
        current_direction = wrist_to_middle.copy()
        current_norm = np.linalg.norm(current_direction)
        target_norm = np.linalg.norm(target_direction)
        
        if current_norm > 1e-6:
            current_direction = current_direction / current_norm
            
            # Calculate cross product to find rotation axis
            cross = np.cross(current_direction, target_direction)
            cross_norm = np.linalg.norm(cross)
            
            # Calculate dot product to find angle
            dot = np.clip(np.dot(current_direction, target_direction), -1.0, 1.0)
            angle = np.arccos(dot)
            
            # If vectors are already aligned, no rotation needed
            if cross_norm > 1e-6 and abs(angle) > 1e-6:
                # Normalize rotation axis
                axis = cross / cross_norm
                
                # Build rotation matrix using Rodrigues' rotation formula
                # R = I + sin(θ) * K + (1 - cos(θ)) * K²
                # where K is the cross-product matrix of the axis
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                
                # Cross-product matrix K
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                
                # Rotation matrix
                I = np.eye(3)
                R = I + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
                
                # Apply rotation to all points
                rotated_points = np.dot(scaled_points, R.T)
                scaled_points = rotated_points
            # If vectors are parallel or anti-parallel, check if we need 180° rotation
            elif cross_norm <= 1e-6 and dot < -0.9:
                # Vectors are opposite - rotate 180° around Z-axis
                R_180 = np.array([
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]
                ])
                scaled_points = np.dot(scaled_points, R_180.T)
    
    # Convert back to list of tuples
    result = [(float(x), float(y), float(z)) for x, y, z in scaled_points]
    
    return result


def normalize_hand_data_with_rotation(landmarks: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """
    Convenience wrapper that always applies rotation alignment.
    """
    return normalize_hand_data(landmarks, apply_rotation=True)


def standardize_features(feature_vector: np.ndarray, scaler: Optional[any] = None, fit: bool = False) -> Tuple[np.ndarray, any]:
    """
    Apply StandardScaler to a feature vector (flattened landmarks + chain code).
    
    This standardizes the features by removing the mean and scaling to unit variance,
    which helps with PCA and classification.
    
    Args:
        feature_vector: 1D numpy array of features (flattened landmarks + chain code)
        scaler: Optional pre-fitted StandardScaler (if None, creates new one)
        fit: If True, fit the scaler on this data; if False, only transform
    
    Returns:
        Tuple of (standardized_features, scaler)
    """
    if not HAS_SKLEARN:
        # If sklearn not available, return features as-is
        return feature_vector, None
    
    if scaler is None:
        scaler = StandardScaler()
    
    # Reshape to 2D array (samples x features) for sklearn
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)
    
    if fit:
        standardized = scaler.fit_transform(feature_vector)
    else:
        standardized = scaler.transform(feature_vector)
    
    # Return as 1D array if input was 1D
    if standardized.shape[0] == 1:
        standardized = standardized.flatten()
    
    return standardized, scaler


def get_hand_info(landmarks_list: List[dict]) -> str:
    """
    Get a string description of which hands were detected.
    
    Args:
        landmarks_list: List of dicts with 'hand' and 'landmarks' keys
    
    Returns:
        String like "left", "right", "both", or "none"
    """
    if not landmarks_list:
        return "none"
    
    hands = [h['hand'] for h in landmarks_list]
    
    if len(hands) == 1:
        return hands[0]
    elif len(hands) == 2:
        return "both"
    else:
        return "unknown"

