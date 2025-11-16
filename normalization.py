"""
Hand landmark normalization module.

This module provides functions to normalize MediaPipe hand landmarks
into a consistent 3D space with configurable spacing and point count.
"""

import numpy as np
from typing import List, Tuple, Optional


def normalize_landmarks_to_3d_space(
    landmarks: List[Tuple[float, float, float]],
    num_points: int,
    spacing_ratio: float = 1.0
) -> List[Tuple[float, float, float]]:
    """
    Normalize hand landmarks into a consistent 3D space.
    
    This function takes raw MediaPipe landmarks (which vary based on
    person size and distance to camera) and normalizes them to a
    consistent coordinate space.
    
    Normalization steps:
    1. Translate to origin (subtract centroid)
    2. Scale to unit size (normalize by maximum distance from origin)
    3. Apply spacing_ratio for overall scaling
    4. Ensure exactly num_points are returned
    
    Args:
        landmarks: List of (x, y, z) tuples from MediaPipe
        num_points: Target number of points to return (should match input length)
        spacing_ratio: Scaling factor for the normalized space (default: 1.0)
                      Values > 1.0 make the space larger, < 1.0 make it smaller
    
    Returns:
        List of normalized (x, y, z) tuples with exactly num_points elements
    """
    if not landmarks:
        # Return zeros if no landmarks provided
        return [(0.0, 0.0, 0.0)] * num_points
    
    # Convert to numpy array for easier manipulation
    points = np.array(landmarks, dtype=np.float32)
    
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
    normalized_points = centered_points / max_distance
    
    # Step 3: Apply spacing ratio for overall scaling
    scaled_points = normalized_points * spacing_ratio
    
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
    spacing_ratio: float = 1.0
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
    
    Returns:
        List of normalized (x, y, z) tuples, or None if no hands detected
    """
    if not landmarks_list:
        return None
    
    # Normalize each hand separately
    normalized_hands = []
    for hand_data in landmarks_list:
        landmarks = hand_data['landmarks']
        normalized = normalize_landmarks_to_3d_space(
            landmarks,
            num_points_per_hand,
            spacing_ratio
        )
        normalized_hands.append(normalized)
    
    # Concatenate all hands into a single list
    all_points = []
    for hand_points in normalized_hands:
        all_points.extend(hand_points)
    
    return all_points


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

