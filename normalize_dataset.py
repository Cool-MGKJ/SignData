"""
Normalize ASL hand landmark dataset for improved ML model performance.

This script applies the following transformations to reduce inter-class distance:
1. Wrist Centering (Translation Invariance)
2. Palm Scaling (Scale Invariance)
3. Rotation Alignment (Y-axis alignment)
4. Z-axis Smoothing
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import glob


# MediaPipe Hand landmark indices
WRIST = 0
MIDDLE_FINGER_BASE = 9  # MCP joint of middle finger
POINTS_PER_HAND = 21


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate 3D Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)


def extract_point(point_dict: Dict) -> np.ndarray:
    """Extract (x, y, z) coordinates from point dictionary."""
    return np.array([point_dict['x'], point_dict['y'], point_dict['z']])


def center_wrist(points: List[np.ndarray]) -> List[np.ndarray]:
    """
    Wrist Centering: Subtract wrist coordinates from all points.
    
    Args:
        points: List of 21 point arrays for a single hand
        
    Returns:
        Centered points (wrist at origin)
    """
    wrist = points[WRIST].copy()
    centered = [pt - wrist for pt in points]
    return centered


def scale_palm(points: List[np.ndarray], use_2d_scale: bool = True) -> Tuple[List[np.ndarray], float]:
    """
    Palm Scaling: Normalize by distance from wrist to middle finger base.
    
    MediaPipe's z-coordinates are relative depth with different units than x,y (which are normalized 0-1).
    To properly account for depth:
    - If use_2d_scale=True: Use 2D (x,y) distance for scaling all coordinates (preserves relative z structure)
    - If use_2d_scale=False: Use 3D distance (original method, but may be skewed by z-scale differences)
    
    Args:
        points: List of 21 centered point arrays for a single hand
        use_2d_scale: If True, use 2D palm distance (recommended for MediaPipe z-depth)
        
    Returns:
        Tuple of (scaled points, scale factor used)
    """
    wrist = points[WRIST]
    middle_base = points[MIDDLE_FINGER_BASE]
    
    if use_2d_scale:
        # Calculate 2D palm scale (distance in x,y plane only)
        # This is better for MediaPipe since z is relative depth with different units
        xy_vector = middle_base[:2] - wrist[:2]  # Only x,y components
        scale = np.linalg.norm(xy_vector)
    else:
        # Original 3D distance (may be affected by z-scale differences)
        scale = euclidean_distance(wrist, middle_base)
    
    # Avoid division by zero
    if scale < 1e-6:
        scale = 1.0
        print(f"Warning: Scale too small ({scale}), using 1.0")
    
    # Scale all coordinates (x, y, z) by the same factor
    # For MediaPipe relative z-depth, this preserves the relative depth relationships
    # while normalizing for hand size based on palm width
    scaled = [pt / scale for pt in points]
    
    return scaled, scale


def rotation_matrix_to_align_y_axis(target_vector: np.ndarray) -> np.ndarray:
    """
    Calculate rotation matrix to align target_vector with positive Y-axis.
    
    Args:
        target_vector: 3D vector to align with Y-axis [0, 1, 0]
        
    Returns:
        3x3 rotation matrix
    """
    y_axis = np.array([0.0, 1.0, 0.0])
    
    # Normalize target vector
    target_norm = np.linalg.norm(target_vector)
    if target_norm < 1e-6:
        # Vector too small, return identity matrix
        return np.eye(3)
    
    target_normalized = target_vector / target_norm
    
    # Calculate rotation axis (cross product)
    v = np.cross(target_normalized, y_axis)
    s = np.linalg.norm(v)  # Sine of angle
    
    if s < 1e-6:
        # Vectors are already aligned (or opposite)
        if np.dot(target_normalized, y_axis) > 0:
            return np.eye(3)  # Already aligned
        else:
            # Opposite direction - rotate 180 degrees around X or Z axis
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    
    # Calculate cosine of angle (dot product of normalized vectors)
    c = np.dot(target_normalized, y_axis)
    
    # Skew-symmetric matrix for cross product
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    # Rodrigues' rotation formula
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    
    return R


def align_rotation(points: List[np.ndarray]) -> List[np.ndarray]:
    """
    Rotation Alignment: Rotate so wrist-to-middle-finger-base aligns with Y-axis.
    
    Args:
        points: List of 21 scaled point arrays for a single hand
        
    Returns:
        Rotated points aligned with Y-axis
    """
    wrist = points[WRIST]
    middle_base = points[MIDDLE_FINGER_BASE]
    
    # Vector from wrist to middle finger base
    wrist_to_middle = middle_base - wrist
    
    # Calculate rotation matrix to align this vector with positive Y-axis
    R = rotation_matrix_to_align_y_axis(wrist_to_middle)
    
    # Apply rotation to all points
    rotated = [R @ pt for pt in points]
    
    return rotated


def smooth_z_axis(points: List[np.ndarray], z_factor: float = 0.8) -> List[np.ndarray]:
    """
    Z-axis Smoothing: Multiply all Z-coordinates by a factor.
    
    MediaPipe provides relative z-depth where:
    - Negative z values indicate points closer to the camera
    - Positive z values indicate points farther from the camera
    - Values are typically in the range [-0.5, 0.5] in normalized space
    
    After wrist centering and palm scaling, we apply a smoothing factor
    to reduce noise while maintaining the relative depth signal.
    
    Args:
        points: List of point arrays (already centered and scaled)
        z_factor: Multiplier for z-coordinates (default: 0.8 for MediaPipe relative depth)
        
    Returns:
        Points with smoothed z-coordinates
    """
    smoothed = []
    for pt in points:
        smoothed_pt = pt.copy()
        # Apply smoothing factor to z-coordinate (preserve sign for MediaPipe relative depth)
        smoothed_pt[2] *= z_factor
        smoothed.append(smoothed_pt)
    return smoothed


def normalize_hand_points(points: List[Dict], apply_rotation: bool = True, z_smoothing_factor: float = 0.8) -> Tuple[List[Dict], Dict]:
    """
    Apply all normalization transformations to a single hand's points.
    
    Note: MediaPipe provides relative z-depth values (not absolute depth measurements).
    Negative z = closer to camera, Positive z = farther from camera.
    After wrist centering and palm scaling, z-values are normalized relative to hand size.
    
    Args:
        points: List of point dictionaries with 'index', 'x', 'y', 'z'
        apply_rotation: Whether to apply rotation alignment (default: True)
        z_smoothing_factor: Factor to multiply z-coordinates (default: 0.8 for MediaPipe relative depth)
                           Set to 1.0 to disable z-axis smoothing
        
    Returns:
        Tuple of (normalized points as dicts, transformation info)
    """
    # Convert to numpy arrays
    point_arrays = [extract_point(pt) for pt in points]
    
    # Step 1: Wrist Centering
    centered = center_wrist(point_arrays)
    
    # Step 2: Palm Scaling
    # Use 2D scale (x,y only) for better MediaPipe z-depth handling
    # MediaPipe z is relative depth with different units, so using 2D palm distance
    # preserves relative z-structure while normalizing for hand size
    scaled, scale_factor = scale_palm(centered, use_2d_scale=True)
    
    # Step 3: Rotation Alignment (optional)
    # Rotation aligns the hand in the x,y plane, but preserves z-depth relationships
    if apply_rotation:
        aligned = align_rotation(scaled)
    else:
        aligned = scaled
    
    # Step 4: Z-axis Smoothing (MediaPipe relative depth smoothing)
    # Apply smoothing factor to compress depth noise while maintaining signal
    # This is applied after all other transformations to reduce z-noise
    normalized = smooth_z_axis(aligned, z_factor=z_smoothing_factor)
    
    # Convert back to dictionary format
    normalized_points = [
        {
            'index': points[i]['index'],
            'x': float(pt[0]),
            'y': float(pt[1]),
            'z': float(pt[2])
        }
        for i, pt in enumerate(normalized)
    ]
    
    # Store transformation metadata
    transform_info = {
        'scale_factor': float(scale_factor),
        'scale_method': '2D_palm_distance_for_MediaPipe',
        'rotation_applied': apply_rotation,
        'z_smoothing_factor': float(z_smoothing_factor),
        'z_depth_source': 'MediaPipe (relative)',
        'depth_handling': 'scaled_by_2D_palm_distance_preserves_relative_z'
    }
    
    return normalized_points, transform_info


def normalize_sample(sample: Dict, apply_rotation: bool = True, z_smoothing_factor: float = 0.8) -> Dict:
    """
    Normalize a single sample (handles both single-hand and double-hand cases).
    
    Args:
        sample: Sample dictionary with 'points' array
        apply_rotation: Whether to apply rotation alignment
        z_smoothing_factor: Factor for z-axis smoothing (default: 0.8 for MediaPipe relative depth)
        
    Returns:
        Normalized sample dictionary
    """
    # Create a copy to avoid modifying original
    normalized_sample = sample.copy()
    points = sample['points']
    
    num_points = len(points)
    
    # Determine if single-hand (21 points) or double-hand (42 points)
    if num_points == POINTS_PER_HAND:
        # Single hand: normalize all 21 points
        normalized_points, transform_info = normalize_hand_points(points, apply_rotation, z_smoothing_factor)
        normalized_sample['points'] = normalized_points
        normalized_sample['normalization_info'] = transform_info
        
        # Update points_flat if it exists
        if 'points_flat' in normalized_sample:
            flattened = []
            for pt in normalized_points:
                flattened.extend([pt['x'], pt['y'], pt['z']])
            normalized_sample['points_flat'] = flattened
            
    elif num_points == POINTS_PER_HAND * 2:
        # Double hand: normalize each hand separately
        hand1_points = points[:POINTS_PER_HAND]
        hand2_points = points[POINTS_PER_HAND:]
        
        normalized_hand1, info1 = normalize_hand_points(hand1_points, apply_rotation, z_smoothing_factor)
        normalized_hand2, info2 = normalize_hand_points(hand2_points, apply_rotation, z_smoothing_factor)
        
        # Preserve original indices (they should already be 0-20 for hand1, 21-41 for hand2)
        # Just combine them
        normalized_points = normalized_hand1 + normalized_hand2
        
        normalized_sample['points'] = normalized_points
        normalized_sample['normalization_info'] = {
            'hand1': info1,
            'hand2': info2
        }
        
        # Update points_flat if it exists
        if 'points_flat' in normalized_sample:
            flattened = []
            for pt in normalized_points:
                flattened.extend([pt['x'], pt['y'], pt['z']])
            normalized_sample['points_flat'] = flattened
            
    else:
        print(f"Warning: Sample {sample.get('id', 'unknown')} has unexpected number of points: {num_points}")
        # Leave points unchanged
        normalized_sample['normalization_info'] = {'error': f'Unexpected point count: {num_points}'}
    
    return normalized_sample


def process_json_file(input_path: Path, output_dir: Path, apply_rotation: bool = True, z_smoothing_factor: float = 0.8) -> int:
    """
    Process a single JSON file and save normalized version.
    
    Args:
        input_path: Path to input JSON file
        output_dir: Directory to save normalized JSON files
        apply_rotation: Whether to apply rotation alignment
        z_smoothing_factor: Factor for z-axis smoothing (default: 0.8 for MediaPipe relative depth)
        
    Returns:
        Number of samples processed
    """
    print(f"Processing: {input_path.name}")
    
    # Load JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if 'samples' in data:
        # Format with metadata and samples array
        samples = data['samples']
        normalized_data = {
            'metadata': data.get('metadata', {}),
            'samples': []
        }
    elif isinstance(data, list):
        # Format with direct samples array
        samples = data
        normalized_data = []
    else:
        print(f"Error: Unknown JSON structure in {input_path.name}")
        return 0
    
    # Normalize each sample
    normalized_samples = []
    for sample in samples:
        try:
            normalized_sample = normalize_sample(sample, apply_rotation, z_smoothing_factor)
            
            if isinstance(normalized_data, list):
                normalized_samples.append(normalized_sample)
            else:
                normalized_data['samples'].append(normalized_sample)
                
        except Exception as e:
            print(f"Error normalizing sample {sample.get('id', 'unknown')}: {e}")
            continue
    
    if isinstance(normalized_data, list):
        normalized_data = normalized_samples
    
    # Update metadata
    if 'metadata' in normalized_data:
        normalized_data['metadata']['normalized'] = True
        normalized_data['metadata']['normalization_timestamp'] = np.datetime64('now').astype(str)
        if isinstance(normalized_data['samples'], list):
            normalized_data['metadata']['total_samples'] = len(normalized_data['samples'])
    
    # Save normalized file
    output_path = output_dir / input_path.name
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)
    
    num_samples = len(normalized_samples) if isinstance(normalized_data, list) else len(normalized_data.get('samples', []))
    print(f"  Normalized {num_samples} samples -> {output_path.name}")
    
    return num_samples


def main():
    """
    Main function to process all JSON files in the data directory.
    
    Configuration:
    - apply_rotation: Apply rotation alignment (recommended)
    - z_smoothing_factor: Factor for z-axis smoothing
      * 0.8 = Apply smoothing (default, good for MediaPipe relative depth)
      * 1.0 = No smoothing (preserve original z-values)
      * < 0.8 = More aggressive smoothing
    """
    # Configuration
    input_dir = Path("data")
    output_dir = Path("normalized_dataset")
    apply_rotation = True  # Set to False to skip rotation alignment
    z_smoothing_factor = 0.8  # MediaPipe relative depth smoothing (1.0 = no smoothing)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Find all JSON files in input directory
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s) to process")
    print("-" * 60)
    
    # Process each JSON file
    total_samples = 0
    for json_file in json_files:
        try:
            num_samples = process_json_file(json_file, output_dir, apply_rotation, z_smoothing_factor)
            total_samples += num_samples
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    print("-" * 60)
    print(f"Processing complete!")
    print(f"Total samples normalized: {total_samples}")
    print(f"Normalized files saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()