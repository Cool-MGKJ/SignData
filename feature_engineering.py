"""
Feature Engineering Module for Hand-Tracking Dataset

This module adds geometric features to hand-tracking data:
- 15 Joint Angles (3D interior angles for each finger: MCP, PIP, DIP)
- Relative Face Distance (dist_to_chin, normalized by palm length)

Features are rotation-invariant and scale-invariant, improving PCA performance.
"""

import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os


def extract_landmark_coords(points: List[Dict[str, Any]], index: int) -> Tuple[float, float, float]:
    """
    Extract (x, y, z) coordinates for a landmark by index.
    
    Args:
        points: List of point dictionaries with 'index', 'x', 'y', 'z' keys
        index: Landmark index (0-20)
    
    Returns:
        Tuple of (x, y, z) coordinates
    """
    for point in points:
        if point['index'] == index:
            return (point['x'], point['y'], point['z'])
    raise ValueError(f"Landmark {index} not found in points")


def calculate_3d_angle(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    p3: Tuple[float, float, float]
) -> float:
    """
    Calculate 3D interior angle at point p2 using dot product/arccosine method.
    
    The angle is formed by vectors (p1->p2) and (p2->p3).
    
    Args:
        p1: First point (x, y, z)
        p2: Vertex point where angle is measured (x, y, z)
        p3: Third point (x, y, z)
    
    Returns:
        Angle in degrees (0-180)
    """
    # Convert to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Calculate vectors from p2 to p1 and p2 to p3
    vec_a = p1 - p2  # Vector from p2 to p1
    vec_b = p3 - p2  # Vector from p2 to p3
    
    # Normalize vectors
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # Avoid division by zero
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    
    vec_a_normalized = vec_a / norm_a
    vec_b_normalized = vec_b / norm_b
    
    # Calculate dot product
    dot_product = np.dot(vec_a_normalized, vec_b_normalized)
    
    # Clamp to [-1, 1] to avoid numerical errors in arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians, then convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_euclidean_distance(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float]
) -> float:
    """
    Calculate 3D Euclidean distance between two points.
    
    Args:
        p1: First point (x, y, z)
        p2: Second point (x, y, z)
    
    Returns:
        Euclidean distance
    """
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)
    return np.linalg.norm(p1_arr - p2_arr)


def calculate_palm_length(points: List[Dict[str, Any]]) -> float:
    """
    Calculate palm length as distance between Wrist (Landmark 0) and Middle Finger MCP (Landmark 9).
    
    Args:
        points: List of point dictionaries
    
    Returns:
        Palm length (Euclidean distance)
    """
    wrist = extract_landmark_coords(points, 0)
    middle_mcp = extract_landmark_coords(points, 9)
    return calculate_euclidean_distance(wrist, middle_mcp)


def calculate_joint_angles(points: List[Dict[str, Any]]) -> List[float]:
    """
    Calculate 15 joint angles (MCP and PIP for each finger, plus DIP for all fingers).
    
    MediaPipe hand landmarks structure:
    - Wrist: 0
    - Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (tip)
    - Index: 5 (MCP), 6 (PIP), 7 (DIP), 8 (tip)
    - Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (tip)
    - Ring: 13 (MCP), 14 (PIP), 15 (DIP), 16 (tip)
    - Pinky: 17 (MCP), 18 (PIP), 19 (DIP), 20 (tip)
    
    For each finger, calculate:
    - MCP angle: angle at MCP joint (e.g., 0-5-6 for index finger)
    - PIP angle: angle at PIP joint (e.g., 5-6-7 for index finger)
    - DIP angle: angle at DIP joint (for completeness, 6-7-8 for index)
    
    This gives 3 angles per finger × 5 fingers = 15 total angles.
    
    Args:
        points: List of point dictionaries with hand landmarks
    
    Returns:
        List of 15 angles in degrees [Index(3), Middle(3), Ring(3), Pinky(3), Thumb(3)]
    """
    angles = []
    
    # Extract all landmarks as (x, y, z) tuples
    landmarks = {}
    for point in points:
        landmarks[point['index']] = (point['x'], point['y'], point['z'])
    
    # Index Finger: MCP (0-5-6), PIP (5-6-7), DIP (6-7-8)
    angles.append(calculate_3d_angle(landmarks[0], landmarks[5], landmarks[6]))  # MCP
    angles.append(calculate_3d_angle(landmarks[5], landmarks[6], landmarks[7]))  # PIP
    angles.append(calculate_3d_angle(landmarks[6], landmarks[7], landmarks[8]))  # DIP
    
    # Middle Finger: MCP (0-9-10), PIP (9-10-11), DIP (10-11-12)
    angles.append(calculate_3d_angle(landmarks[0], landmarks[9], landmarks[10]))  # MCP
    angles.append(calculate_3d_angle(landmarks[9], landmarks[10], landmarks[11]))  # PIP
    angles.append(calculate_3d_angle(landmarks[10], landmarks[11], landmarks[12]))  # DIP
    
    # Ring Finger: MCP (0-13-14), PIP (13-14-15), DIP (14-15-16)
    angles.append(calculate_3d_angle(landmarks[0], landmarks[13], landmarks[14]))  # MCP
    angles.append(calculate_3d_angle(landmarks[13], landmarks[14], landmarks[15]))  # PIP
    angles.append(calculate_3d_angle(landmarks[14], landmarks[15], landmarks[16]))  # DIP
    
    # Pinky Finger: MCP (0-17-18), PIP (17-18-19), DIP (18-19-20)
    angles.append(calculate_3d_angle(landmarks[0], landmarks[17], landmarks[18]))  # MCP
    angles.append(calculate_3d_angle(landmarks[17], landmarks[18], landmarks[19]))  # PIP
    angles.append(calculate_3d_angle(landmarks[18], landmarks[19], landmarks[20]))  # DIP
    
    # Thumb: MCP (0-1-2), IP (1-2-3), DIP (2-3-4)
    # Note: Thumb has different structure (CMC, MCP, IP, tip)
    angles.append(calculate_3d_angle(landmarks[0], landmarks[1], landmarks[2]))  # MCP
    angles.append(calculate_3d_angle(landmarks[1], landmarks[2], landmarks[3]))   # IP (equivalent to PIP)
    angles.append(calculate_3d_angle(landmarks[2], landmarks[3], landmarks[4]))   # DIP
    
    return angles


def calculate_dist_to_chin(
    points: List[Dict[str, Any]],
    face_landmarks: Optional[Dict[int, Tuple[float, float, float]]] = None
) -> Optional[float]:
    """
    Calculate distance from Index Fingertip (Landmark 8) to Chin (Face Landmark 152).
    The distance is normalized by palm length for scale invariance.
    
    Args:
        points: List of point dictionaries with hand landmarks
        face_landmarks: Optional dictionary mapping face landmark indices to (x, y, z) tuples
    
    Returns:
        Normalized distance to chin, or None if face_landmarks not provided
    """
    if face_landmarks is None or 152 not in face_landmarks:
        return None
    
    # Extract Index Fingertip (Landmark 8)
    index_tip = extract_landmark_coords(points, 8)
    
    # Extract Chin (Face Landmark 152)
    chin = face_landmarks[152]
    
    # Calculate raw distance
    raw_distance = calculate_euclidean_distance(index_tip, chin)
    
    # Normalize by palm length
    palm_length = calculate_palm_length(points)
    
    if palm_length < 1e-6:
        return None  # Invalid palm length
    
    normalized_distance = raw_distance / palm_length
    
    return normalized_distance


def process_sample(
    sample: Dict[str, Any],
    face_landmarks: Optional[Dict[int, Tuple[float, float, float]]] = None
) -> Dict[str, Any]:
    """
    Process a single sample to add engineered features.
    
    Args:
        sample: Sample dictionary with 'points' key
        face_landmarks: Optional dictionary of face landmarks
    
    Returns:
        Sample dictionary with added 'engineered_features' key
    """
    points = sample['points']
    
    # Calculate 15 joint angles
    joint_angles = calculate_joint_angles(points)
    
    # Calculate dist_to_chin (normalized by palm length)
    dist_to_chin = calculate_dist_to_chin(points, face_landmarks)
    
    # Create engineered_features array
    engineered_features = joint_angles.copy()
    
    if dist_to_chin is not None:
        engineered_features.append(dist_to_chin)
    else:
        # If no face landmarks, append 0.0 as placeholder
        engineered_features.append(0.0)
    
    # Add to sample
    sample['engineered_features'] = engineered_features
    
    return sample


def process_json_file(
    input_path: str,
    output_path: Optional[str] = None,
    face_landmarks: Optional[Dict[int, Tuple[float, float, float]]] = None
) -> None:
    """
    Process a JSON file to add engineered features to all samples.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (default: input_path with '_enhanced' suffix)
        face_landmarks: Optional dictionary of face landmarks
    """
    # Load input data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each sample
    processed_samples = []
    for sample in data['samples']:
        processed_sample = process_sample(sample, face_landmarks)
        processed_samples.append(processed_sample)
    
    # Update data structure
    data['samples'] = processed_samples
    
    # Update metadata
    if 'metadata' not in data:
        data['metadata'] = {}
    data['metadata']['features_added'] = {
        'joint_angles': 15,
        'dist_to_chin': 1,
        'total_features': 16,
        'description': '15 joint angles (MCP, PIP, DIP for each finger) + 1 normalized dist_to_chin'
    }
    
    # Determine output path
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_enhanced.json"
    
    # Save output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Processed {len(processed_samples)} samples")
    print(f"[OK] Output saved to: {output_path}")
    print(f"[OK] Features added: 15 joint angles + 1 dist_to_chin = 16 total features")
    print(f"  - Joint angles are rotation-invariant")
    print(f"  - dist_to_chin is normalized by palm length (scale-invariant)")


def main():
    """
    Main function to process the dataset.
    """
    # Example usage
    input_file = "data/asl_dataset.json"
    
    # Optional: If you have face landmarks, provide them as a dictionary
    # face_landmarks_example = {
    #     152: (x, y, z)  # Chin coordinates
    # }
    
    # Process without face landmarks (dist_to_chin will be 0.0)
    process_json_file(input_file, face_landmarks=None)
    
    # Or process with face landmarks:
    # process_json_file(input_file, face_landmarks=face_landmarks_example)


if __name__ == "__main__":
    main()

