"""
Face-centered 3D voxel grid using MediaPipe FaceMesh (MediaPipe z-depth only).

This module constructs a 3D voxel cube (rectangular prism) centered on the face
using MediaPipe FaceMesh for landmark detection. It records multi-layer voxel
hits and ordered voxel travel paths per landmark.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict


# MediaPipe Hand landmark indices (for path tracking)
INDEX_TIP = 8
THUMB_TIP = 4
LANDMARK_NAMES = {
    0: "wrist",
    1: "thumb_cmc", 2: "thumb_mcp", 3: "thumb_ip", 4: "thumb_tip",
    5: "index_mcp", 6: "index_pip", 10: "index_dip", 8: "index_tip",
    9: "middle_mcp", 10: "middle_pip", 11: "middle_dip", 8: "middle_tip",
    13: "ring_mcp", 14: "ring_pip", 10: "ring_dip", 16: "ring_tip",
    110: "pinky_mcp", 18: "pinky_pip", 19: "pinky_dip", 20: "pinky_tip"
}

# 26-directional chain code constants (3D unit vectors)
# 6 face directions (orthogonal): ±x, ±y, ±z
# 12 edge directions (diagonal on one plane): ±x±y, ±x±z, ±y±z (no zero)
# 8 corner directions (diagonal in all 3 axes): ±x±y±z (all non-zero)
DIRECTION_26 = np.array([
    # 6 face directions (0-5)
    [ 1,  0,  0],  # 0: +x
    [-1,  0,  0],  # 1: -x
    [ 0,  1,  0],  # 2: +y
    [ 0, -1,  0],  # 3: -y
    [ 0,  0,  1],  # 4: +z
    [ 0,  0, -1],  # 5: -z
    # 12 edge directions (6-17)
    [ 1,  1,  0],  # 6: +x+y
    [ 1, -1,  0],  # 7: +x-y
    [-1,  1,  0],  # 8: -x+y
    [-1, -1,  0],  # 9: -x-y
    [ 1,  0,  1],  # 10: +x+z
    [ 1,  0, -1],  # 11: +x-z
    [-1,  0,  1],  # 12: -x+z
    [-1,  0, -1],  # 13: -x-z
    [ 0,  1,  1],  # 14: +y+z
    [ 0,  1, -1],  # 15: +y-z
    [ 0, -1,  1],  # 16: -y+z
    [ 0, -1, -1],  # 17: -y-z
    # 8 corner directions (18-25)
    [ 1,  1,  1],  # 18: +x+y+z
    [ 1,  1, -1],  # 19: +x+y-z
    [ 1, -1,  1],  # 20: +x-y+z
    [ 1, -1, -1],  # 21: +x-y-z
    [-1,  1,  1],  # 22: -x+y+z
    [-1,  1, -1],  # 23: -x+y-z
    [-1, -1,  1],  # 24: -x-y+z
    [-1, -1, -1],  # 25: -x-y-z
], dtype=np.float32)

# Normalize all direction vectors to unit length
for i in range(len(DIRECTION_26)):
    norm = np.linalg.norm(DIRECTION_26[i])
    if norm > 0:
        DIRECTION_26[i] = DIRECTION_26[i] / norm


class FaceGrid3D:
    """
    Tracks a face-centered 3D voxel grid using MediaPipe FaceMesh z-depth.
    
    **2-Layer System (Face Layer + Forward Layer):**
    - Layer 0 "Face Layer" (z_idx=0, voxels 0-79): Hand at or behind face plane (relative_z >= 0)
    - Layer 1 "Forward Layer" (z_idx=1, voxels 80-159): Hand extended toward camera (relative_z < 0)
    
    **Z-Axis Convention (MediaPipe):**
    - Negative z = closer to camera
    - Positive z = farther from camera
    - relative_z = (raw_z - face_z_reference) / reference_length
    - negative relative_z → hand extended forward → Layer 1 (near camera)
    - positive relative_z → hand near/behind face → Layer 0 (near face)
    
    **Voxel Indexing:**
    - Order: [z, row, col] - z (depth layer) is fastest, then row (y), then col (x)
    - Formula: idx = z_idx * (breadth * length) + y_idx * breadth + x_idx
    - Default: 8 wide × 10 tall × 2 deep = 160 voxels total
    
    **Dynamic Head Tracking:**
    - Grid is centered on nose and rotates with head yaw/pitch
    - Ensures layers are always positioned relative to user's body
    
    **Data Collected Per Sample:**
    - Normalized hand landmarks (normalized in real-time during capture)
    - Hit order: sequence of voxel indices touched during gesture
    - Chain code: directional trajectory (0-25 direction indices)
    """
    
    # MediaPipe Face Mesh landmark indices
    NOSE_TIP = 4  # Nose tip (stable point)
    LEFT_EYE_INNER = 133  # Left eye inner corner
    LEFT_EYE_OUTER = 33  # Left eye outer corner
    RIGHT_EYE_INNER = 362  # Right eye inner corner
    RIGHT_EYE_OUTER = 263  # Right eye outer corner
    
    def __init__(
        self,
        breadth: int = 8,  # x-axis (width)
        length: int = 10,   # y-axis (height)
        depth_layers: int = 3,  # z-axis (depth, 3 layers: near, middle, far)
        depth_span_factor: float = 2.4,  # How many spacing steps span forward/back
        hit_radius_norm: float = 0.12,  # Hit detection radius in normalized 3D space (increased for better triggering)
        track_landmark_paths: bool = True,
        tracked_landmarks: Optional[List[int]] = None
    ):
        """
        Initialize the 3D voxel grid tracker.
        
        Args:
            breadth: Number of voxels horizontally (x-axis, default: 8)
            length: Number of voxels vertically (y-axis, default: 10)
            depth_layers: Number of depth layers (z-axis, default: 3)
            depth_span_factor: Multiplier for depth layer span (default: 2.4)
            hit_radius_norm: Hit detection radius in normalized 3D coordinates (default: 0.06)
            track_landmark_paths: If True, record ordered paths per landmark (default: True)
            tracked_landmarks: List of landmark indices to track paths for (default: [8, 4] = index_tip, thumb_tip)
        """
        self.breadth = breadth
        self.length = length
        self.depth_layers = depth_layers
        self.num_voxels = breadth * length * depth_layers
        self.depth_span_factor = depth_span_factor
        self.hit_radius_norm = hit_radius_norm
        
        # Path tracking configuration
        self.track_landmark_paths = track_landmark_paths
        if tracked_landmarks is None:
            self.tracked_landmarks = [INDEX_TIP, THUMB_TIP]  # index_tip, thumb_tip
        else:
            self.tracked_landmarks = tracked_landmarks
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize MediaPipe Pose for shoulder detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Grid state
        self.voxel_centers = None  # List of (x, y, z) tuples in normalized coordinates
        self.voxel_centers_2d = None  # List of (u, v) tuples in pixel coordinates (for visualization)
        self.voxel_hit_bool = None  # Flattened boolean array [num_voxels]
        self.voxel_hit_order = []  # Ordered voxel IDs as they are hit
        self.voxel_paths = None  # Dict mapping landmark_idx -> list of voxel indices (ordered)
        self.voxel_last_index_per_landmark = None  # Dict mapping landmark_idx -> last voxel index
        self.frame_width = None
        self.frame_height = None
        self.grid_center_z = 0.0  # Depth at nose pixel (MediaPipe z normalized)
        self.grid_spacing_norm = 0.0  # Normalized spacing used
        self.depth_layer_thickness = 0.0  # Δz between depth layers
        self.max_grid_yaw_deg = 40.0  # Clamp yaw rotation
        self.max_grid_pitch_deg = 30.0  # Clamp pitch rotation
        self.head_yaw_deg = 0.0
        self.head_pitch_deg = 0.0
        
        # Store current trigger point for visualization
        self.current_trigger_point_2d = None  # (u, v) pixel coordinates
        self.current_trigger_point_3d = None  # (x, y, z) normalized coordinates
        
        # Layer z-position tracking (continuously updated relative to face)
        self.layer_z_positions = None  # List of z-positions relative to face, one per layer [z_layer0, z_layer1, ...]
        self.initial_trigger_z = None  # Initial trigger z-position when capture starts
        self.z_matching_tolerance = 0.25  # Tolerance for z-coordinate matching (increased to allow 2nd layer triggering)
        
        # Dynamic depth scaling system
        self.reference_length = None  # Shoulder-to-shoulder or wrist-to-elbow distance
        self.face_z_reference = None  # Face plane Z reference (nose or mid-shoulders Z)
        self.last_pose_results = None  # Store last pose detection results
        
        # 26-directional chain code validation system
        self.current_gesture_chain = []  # List of direction indices (0-25) representing movement trajectory
        self.jitter_threshold = 0.02  # Minimum movement distance to record a direction change
        self.is_capturing_chain = False  # Flag to track if chain code should be updated (only during capture)
    
    def reset_hit_tracking(self):
        """Reset hit tracking and paths (call at start of new capture)."""
        self.voxel_hit_bool = np.zeros(self.num_voxels, dtype=bool)
        self.voxel_paths = {landmark_idx: [] for landmark_idx in self.tracked_landmarks}
        self.voxel_last_index_per_landmark = {landmark_idx: -1 for landmark_idx in self.tracked_landmarks}
        self.voxel_hit_order = []
        self.current_trigger_point_2d = None
        self.current_trigger_point_3d = None
        # Reset layer z-positions (will be captured when capture starts)
        self.layer_z_positions = None
        self.initial_trigger_z = None
        # Reset dynamic depth scaling (will be recalculated)
        self.reference_length = None
        self.face_z_reference = None
        # Reset chain code tracking
        self.current_gesture_chain = []
        self.is_capturing_chain = False
    
    def start_chain_capture(self):
        """Start capturing chain code (call when capture starts)."""
        self.is_capturing_chain = True
        self.current_gesture_chain = []  # Reset chain code at start of capture
    
    def stop_chain_capture(self):
        """Stop capturing chain code (call when capture stops)."""
        self.is_capturing_chain = False
    
    def update_layer_z_positions_relative_to_face(self):
        """
        Continuously update layer z-positions relative to current face depth.
        This calculates the z-offset of each layer from the current face/nose depth.
        The layer z-positions are stored as offsets from face depth, making them
        always relative to the face position.
        
        This should be called whenever the grid is updated (in process_frame).
        """
        if self.voxel_centers is None:
            return
        
        # Get current face depth (nose depth)
        face_depth = self.grid_center_z
        if face_depth is None or face_depth <= 0:
            face_depth = 0.5  # Default if not set
        
        # Calculate average z-position for each layer, then convert to relative offsets
        voxels_per_layer = self.breadth * self.length
        layer_z_offsets = []  # Store as offsets from face depth
        
        for z_idx in range(self.depth_layers):
            # Get all voxels in this layer
            layer_start_idx = z_idx * voxels_per_layer
            layer_end_idx = layer_start_idx + voxels_per_layer
            
            # Calculate average z-position of voxels in this layer
            layer_voxels = self.voxel_centers[layer_start_idx:layer_end_idx]
            if layer_voxels:
                avg_z = np.mean([vz for _, _, vz in layer_voxels])
                # Calculate offset relative to face depth
                z_offset = avg_z - face_depth
                layer_z_offsets.append(z_offset)
            else:
                layer_z_offsets.append(0.0)  # Default offset if no voxels
        
        # Store as relative offsets from face depth
        self.layer_z_positions = layer_z_offsets
    
    def capture_layer_z_positions(self, trigger_point_z: Optional[float] = None):
        """
        Capture z-positions of each layer (legacy method, calls update_layer_z_positions_relative_to_face).
        This is kept for backward compatibility.
        
        Args:
            trigger_point_z: Initial trigger point z-position (optional, can be captured later)
        """
        # Use the new method that updates relative to face
        self.update_layer_z_positions_relative_to_face()
        
        # Store initial trigger z if provided
        if trigger_point_z is not None:
            self.initial_trigger_z = trigger_point_z
        
        # Calculate absolute z-positions for debug output
        face_depth = self.grid_center_z if self.grid_center_z is not None else 0.5
        absolute_z_positions = [face_depth + offset for offset in self.layer_z_positions]
        print(f"Updated layer z-positions relative to face (face_depth={face_depth:.3f}): {absolute_z_positions}")
        if self.initial_trigger_z is not None:
            print(f"Initial trigger z: {self.initial_trigger_z}")
    
    def voxel_index(self, x_idx: int, y_idx: int, z_idx: int) -> int:
        """
        Convert 3D voxel indices to linear index.
        
        Indexing order: [z, row, col] - z (depth layer) is fastest, then row (y), then col (x)
        Formula: idx = z * (breadth * length) + y * breadth + x
        
        Args:
            x_idx: Column index (0 to breadth-1)
            y_idx: Row index (0 to length-1)
            z_idx: Depth layer index (0 to depth_layers-1)
            
        Returns:
            Linear voxel index (0 to num_voxels-1)
        """
        return z_idx * (self.breadth * self.length) + y_idx * self.breadth + x_idx
    
    def get_voxel_centers(self) -> List[Tuple[float, float, float]]:
        """
        Get all voxel center coordinates.
        
        Returns:
            List of (x, y, z) tuples in normalized coordinates
            Order: [z, row, col] - z fastest, then row, then col
        """
        if self.voxel_centers is None:
            return []
        return self.voxel_centers.copy()
    
    def project_to_pixels(self, point: Tuple[float, float, float]) -> Tuple[int, int]:
        """
        Project a 3D normalized point to pixel coordinates.
        
        Args:
            point: (x_norm, y_norm, z_norm) in normalized coordinates
            
        Returns:
            (u, v) pixel coordinates
        """
        x_norm, y_norm, _ = point
        u = int(x_norm * self.frame_width)
        v = int(y_norm * self.frame_height)
        return (u, v)
    
    def get_hit_grid_3d(self) -> List[int]:
        """
        Get the hit grid as a flattened 1D vector of 0s and 1s.
        
        Returns:
            List of integers (0 or 1) of length breadth * length * depth_layers
            Order: [z, row, col] - z fastest, then row, then col
        """
        if self.voxel_hit_bool is None:
            return [0] * self.num_voxels
        
        return [1 if self.voxel_hit_bool[i] else 0 for i in range(self.num_voxels)]
    
    def get_voxel_paths(self) -> Dict[str, List[int]]:
        """
        Get ordered voxel paths per landmark.
        
        Returns:
            Dictionary mapping landmark names to ordered lists of voxel indices
        """
        if self.voxel_paths is None:
            return {}
        
        result = {}
        for landmark_idx, path in self.voxel_paths.items():
            landmark_name = LANDMARK_NAMES.get(landmark_idx, f"landmark_{landmark_idx}")
            result[landmark_name] = path.copy()
        
        return result
    
    def get_voxel_hit_centers(self) -> List[Tuple[float, float, float]]:
        """
        Get the 3D coordinates of all hit voxel centers.
        
        Returns:
            List of (x, y, z) tuples for each voxel that was hit
        """
        if self.voxel_hit_bool is None or self.voxel_centers is None:
            return []
        
        hit_centers = []
        for i in range(self.num_voxels):
            if self.voxel_hit_bool[i]:
                hit_centers.append(self.voxel_centers[i])
        
        return hit_centers
    
    def get_hit_order(self) -> List[int]:
        """
        Get the ordered list of voxel indices representing the hit sequence.
        """
        return self.voxel_hit_order.copy() if self.voxel_hit_order else []
    
    def get_chain_code(self) -> List[int]:
        """
        Get the current gesture chain code (sequence of direction indices 0-25).
        """
        return self.current_gesture_chain.copy()
    
    def _update_chain_code(self, voxel_idx: int):
        """
        Update the chain code when a new voxel is hit.
        
        Calculates the 3D direction vector from the previous voxel to the current one,
        finds the closest match among the 26 directions using cosine similarity,
        and appends the direction index to the chain code.
        
        Args:
            voxel_idx: Index of the newly hit voxel
        """
        # Only update chain code during capture
        if not self.is_capturing_chain:
            return
            
        if self.voxel_centers is None or voxel_idx >= len(self.voxel_centers):
            return
        
        # Need at least 2 points to calculate a direction
        if len(self.voxel_hit_order) < 2:
            return
        
        # Get the previous voxel index (second to last in the order)
        prev_voxel_idx = self.voxel_hit_order[-2]
        
        # Get 3D positions of previous and current voxels
        prev_x, prev_y, prev_z = self.voxel_centers[prev_voxel_idx]
        curr_x, curr_y, curr_z = self.voxel_centers[voxel_idx]
        
        # Calculate direction vector
        direction_vec = np.array([
            curr_x - prev_x,
            curr_y - prev_y,
            curr_z - prev_z
        ], dtype=np.float32)
        
        # Check if movement exceeds jitter threshold
        movement_distance = np.linalg.norm(direction_vec)
        if movement_distance < self.jitter_threshold:
            return  # Movement too small, ignore
        
        # Normalize the direction vector
        if movement_distance > 0:
            direction_vec = direction_vec / movement_distance
        
        # Find the closest match among the 26 directions using cosine similarity
        # Cosine similarity: cos(θ) = dot(v1, v2) / (|v1| * |v2|)
        # Since both are normalized, cos(θ) = dot(v1, v2)
        # Higher cosine similarity = more similar direction
        max_similarity = -1.0
        best_direction_idx = 0
        
        for idx, dir_vec in enumerate(DIRECTION_26):
            # Both vectors are normalized, so cosine similarity is just the dot product
            similarity = np.dot(direction_vec, dir_vec)
            if similarity > max_similarity:
                max_similarity = similarity
                best_direction_idx = idx
        
        # Only append if it's different from the last recorded direction
        # This handles variable movement speeds by not duplicating consecutive directions
        if len(self.current_gesture_chain) == 0 or self.current_gesture_chain[-1] != best_direction_idx:
            self.current_gesture_chain.append(best_direction_idx)
    

    
    def calculate_reference_length_from_pose(self, pose_results) -> Optional[float]:
        """
        Calculate reference length from pose landmarks (shoulder-to-shoulder or wrist-to-elbow).
        
        Args:
            pose_results: MediaPipe Pose results
            
        Returns:
            Reference length in normalized 3D space, or None if not available
        """
        if pose_results is None or not pose_results.pose_landmarks:
            return None
        
        landmarks = pose_results.pose_landmarks.landmark
        
        # MediaPipe Pose landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_WRIST = 15
        LEFT_ELBOW = 13
        RIGHT_WRIST = 16
        RIGHT_ELBOW = 14
        
        # Try shoulder-to-shoulder first (preferred)
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        
        if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5):
            # Calculate 3D Euclidean distance between shoulders
            shoulder_distance = np.sqrt(
                (left_shoulder.x - right_shoulder.x) ** 2 +
                (left_shoulder.y - right_shoulder.y) ** 2 +
                (left_shoulder.z - right_shoulder.z) ** 2
            )
            if shoulder_distance > 0.01:  # Valid distance
                return shoulder_distance
        
        # Fallback: Use wrist-to-elbow distance (try left first, then right)
        left_wrist = landmarks[LEFT_WRIST]
        left_elbow = landmarks[LEFT_ELBOW]
        right_wrist = landmarks[RIGHT_WRIST]
        right_elbow = landmarks[RIGHT_ELBOW]
        
        # Try left arm
        if (left_wrist.visibility > 0.5 and left_elbow.visibility > 0.5):
            wrist_elbow_distance = np.sqrt(
                (left_wrist.x - left_elbow.x) ** 2 +
                (left_wrist.y - left_elbow.y) ** 2 +
                (left_wrist.z - left_elbow.z) ** 2
            )
            if wrist_elbow_distance > 0.01:  # Valid distance
                return wrist_elbow_distance
        
        # Try right arm
        if (right_wrist.visibility > 0.5 and right_elbow.visibility > 0.5):
            wrist_elbow_distance = np.sqrt(
                (right_wrist.x - right_elbow.x) ** 2 +
                (right_wrist.y - right_elbow.y) ** 2 +
                (right_wrist.z - right_elbow.z) ** 2
            )
            if wrist_elbow_distance > 0.01:  # Valid distance
                return wrist_elbow_distance
        
        return None
    
    def calculate_face_z_reference(self, face_landmarks) -> Optional[float]:
        """
        Calculate face Z reference (zero-point calibration) from nose or mid-shoulders.
        
        Args:
            face_landmarks: MediaPipe Face Mesh landmarks
            
        Returns:
            Face Z reference value (MediaPipe z coordinate), or None if not available
        """
        if face_landmarks is None:
            return None
        
        # Use nose tip Z as face plane reference
        nose_tip = face_landmarks.landmark[self.NOSE_TIP]
        return nose_tip.z
    
    def calculate_relative_z(self, raw_z: float) -> Optional[float]:
        """
        Transform raw Z value to relative Z using dynamic depth scaling.
        
        Formula: Relative_Z = (Raw_Z - Face_Z_Reference) / Reference_Length
        
        Args:
            raw_z: Raw MediaPipe Z coordinate
            
        Returns:
            Relative Z value, or None if reference values not available
        """
        if self.face_z_reference is None or self.reference_length is None:
            return None
        
        if self.reference_length <= 0:
            return None
        
        relative_z = (raw_z - self.face_z_reference) / self.reference_length
        return relative_z
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Process a frame to detect face and build/update the 3D voxel grid
        using MediaPipe z-depth only (no MiDaS).
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            True if face detected and grid built, False otherwise
        """
        if self.face_mesh is None:
            return False
        
        # Store frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process with Pose for reference length calculation
        if self.pose is not None:
            pose_results = self.pose.process(rgb_frame)
            self.last_pose_results = pose_results
            
            # Calculate reference length from pose
            ref_length = self.calculate_reference_length_from_pose(pose_results)
            if ref_length is not None:
                self.reference_length = ref_length
        
        # Process with Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
            return False
        
        # Get the first (and only) face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract key landmarks
        nose_tip = face_landmarks.landmark[self.NOSE_TIP]
        left_eye_inner = face_landmarks.landmark[self.LEFT_EYE_INNER]
        left_eye_outer = face_landmarks.landmark[self.LEFT_EYE_OUTER]
        right_eye_inner = face_landmarks.landmark[self.RIGHT_EYE_INNER]
        right_eye_outer = face_landmarks.landmark[self.RIGHT_EYE_OUTER]
        
        # Calculate eye centers
        left_eye_center = (
            (left_eye_inner.x + left_eye_outer.x) / 2,
            (left_eye_inner.y + left_eye_outer.y) / 2
        )
        right_eye_center = (
            (right_eye_inner.x + right_eye_outer.x) / 2,
            (right_eye_inner.y + right_eye_outer.y) / 2
        )
        left_eye_center_3d = (
            (left_eye_inner.x + left_eye_outer.x) / 2,
            (left_eye_inner.y + left_eye_outer.y) / 2,
            (left_eye_inner.z + left_eye_outer.z) / 2
        )
        right_eye_center_3d = (
            (right_eye_inner.x + right_eye_outer.x) / 2,
            (right_eye_inner.y + right_eye_outer.y) / 2,
            (right_eye_inner.z + right_eye_outer.z) / 2
        )
        
        forehead = face_landmarks.landmark[10]
        # Use landmark 152 for chin (standard MediaPipe Face Mesh index)
        chin = face_landmarks.landmark[152]
        
        # Nose center (using nose tip) - this updates with head movement in 3D
        nose_center = (nose_tip.x, nose_tip.y)
        nose_center_z_mp = nose_tip.z  # MediaPipe z (relative depth)
        
        # Get depth at nose pixel using MediaPipe z only (normalized)
        # MediaPipe z is relative and can be negative, so we normalize it.
        # Map typical MediaPipe z range (-0.5 to 0.5) to 0-1.
        # This depth value updates with head movement forward/backward
        nose_depth = np.clip((nose_center_z_mp + 0.5) / 1.0, 0.0, 1.0)
        if nose_depth < 0.01:  # If too close to 0, use a default
            nose_depth = 0.5
        
        # Store grid center z for reference - this follows head depth movement
        self.grid_center_z = nose_depth
        
        # Calculate and store face Z reference (zero-point calibration)
        face_z_ref = self.calculate_face_z_reference(face_landmarks)
        if face_z_ref is not None:
            self.face_z_reference = face_z_ref
        
        # Calculate horizontal spacing unit (distance from nose to left eye)
        # Use 3D distance to account for head movement in depth
        dx_norm = np.sqrt(
            (nose_center[0] - left_eye_center[0]) ** 2 +
            (nose_center[1] - left_eye_center[1]) ** 2
        )
        
        # Don't scale by depth - keep grid size constant relative to head size
        # This ensures the grid moves with the head in 3D space
        if dx_norm < 1e-6:
            dx_norm = 0.05  # Default spacing
        
        # Calculate vertical spacing
        eye_distance = np.sqrt(
            (left_eye_center[0] - right_eye_center[0]) ** 2 +
            (left_eye_center[1] - right_eye_center[1]) ** 2
        )
        dy_norm = eye_distance * 1.2
        
        self.grid_spacing_norm = (dx_norm + dy_norm) / 2
        
        # Calculate depth layer thickness (Δz)
        layer_divisor = max(self.depth_layers - 1, 1)
        dz_norm = dx_norm * self.depth_span_factor / layer_divisor
        self.depth_layer_thickness = dz_norm
        
        # Estimate head yaw (rotation around vertical axis) using 3D eye vector
        eye_vector_3d = np.array([
            right_eye_center_3d[0] - left_eye_center_3d[0],
            right_eye_center_3d[1] - left_eye_center_3d[1],
            right_eye_center_3d[2] - left_eye_center_3d[2]
        ])
        yaw_rad = np.arctan2(eye_vector_3d[2], eye_vector_3d[0] + 1e-6)
        yaw_rad = np.clip(
            yaw_rad,
            np.radians(-self.max_grid_yaw_deg),
            np.radians(self.max_grid_yaw_deg)
        )
        self.head_yaw_deg = np.degrees(yaw_rad)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        
        # Estimate head pitch (rotation around horizontal axis) using forehead-chin vector
        # When nodding down: chin.y increases (down), chin.z increases (forward)
        # In MediaPipe: y increases downward, z increases forward
        vertical_vector_3d = np.array([
            chin.x - forehead.x,
            chin.y - forehead.y,
            chin.z - forehead.z
        ])
        # Calculate pitch angle: arctan2(dz, dy) gives angle from vertical
        # When nodding down: dy > 0 (chin down), dz > 0 (chin forward)
        # arctan2(dz, dy) gives positive angle when both are positive (nodding down)
        pitch_rad = np.arctan2(vertical_vector_3d[2], vertical_vector_3d[1] + 1e-6)
        # Invert the pitch to fix far layer moving opposite - when nodding down, 
        # we want both layers to move down together
        pitch_rad = -pitch_rad
        pitch_rad = np.clip(
            pitch_rad,
            np.radians(-self.max_grid_pitch_deg),
            np.radians(self.max_grid_pitch_deg)
        )
        self.head_pitch_deg = np.degrees(pitch_rad)
        cos_pitch = np.cos(pitch_rad)
        sin_pitch = np.sin(pitch_rad)
        
        # Build the 3D voxel grid
        voxel_centers = []
        voxel_centers_2d = []
        
        half_breadth = (self.breadth - 1) / 2
        half_depth = (self.depth_layers - 1) / 2
        top_y_norm = max(forehead.y - dy_norm, 0.0)
        
        # Order: [z, row, col] - z fastest, then row, then col.
        # z_idx=0 is the layer closest to the face (near), higher z_idx is farther.
        for z_idx in range(self.depth_layers):
            for y_idx in range(self.length):
                for x_idx in range(self.breadth):
                    # Calculate offset from center
                    offset_x = (x_idx - half_breadth) * dx_norm
                    base_y = top_y_norm + y_idx * dy_norm
                    base_y_clamped = np.clip(base_y, 0.0, 1.0)
                    offset_y = base_y_clamped - nose_center[1]
                    # z_idx 0: near layer at nose_depth; higher z_idx: farther layers
                    # Each layer is spaced by dz_norm
                    offset_z = z_idx * dz_norm
                    
                    # Rotate offsets around the vertical axis (yaw) first
                    rotated_x = offset_x * cos_yaw + offset_z * sin_yaw
                    rotated_z_after_yaw = -offset_x * sin_yaw + offset_z * cos_yaw
                    
                    # Rotate around horizontal axis (pitch) after yaw rotation
                    # When nodding down, both layers should tilt down together around the nose
                    # We've inverted pitch_rad above, so when nodding down, pitch_rad is negative
                    # Standard rotation matrix around x-axis (applied with inverted pitch):
                    #   [1    0        0    ]
                    #   [0  cos(p) -sin(p) ]
                    #   [0  sin(p)  cos(p) ]
                    # With negative pitch (nodding down): sin(p) < 0, so far layer moves down correctly
                    rotated_y = offset_y * cos_pitch - rotated_z_after_yaw * sin_pitch
                    rotated_z_pitch = offset_y * sin_pitch + rotated_z_after_yaw * cos_pitch
                    
                    # Create voxel center in normalized space (add rotated offsets to nose center)
                    voxel_x = nose_center[0] + rotated_x
                    voxel_y = np.clip(nose_center[1] + rotated_y, 0.0, 1.0)
                    voxel_z = nose_depth + rotated_z_pitch  # Relative to nose depth
                    
                    # Clamp x, y to valid range [0, 1]
                    voxel_x = np.clip(voxel_x, 0.0, 1.0)
                    voxel_y = np.clip(voxel_y, 0.0, 1.0)
                    # z can be outside [0, 1] but we'll normalize it
                    voxel_z = np.clip(voxel_z, 0.0, 1.0)
                    
                    voxel_centers.append((voxel_x, voxel_y, voxel_z))
                    
                    # Convert to pixel coordinates for drawing
                    pixel_u = int(voxel_x * self.frame_width)
                    pixel_v = int(voxel_y * self.frame_height)
                    voxel_centers_2d.append((pixel_u, pixel_v))
        
        self.voxel_centers = voxel_centers
        self.voxel_centers_2d = voxel_centers_2d
        
        # Continuously update layer z-positions relative to current face depth
        # This ensures layers are always positioned relative to the face
        self.update_layer_z_positions_relative_to_face()
        
        return True
    
    def calculate_palm_trigger_point(self, landmarks: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        """
        Calculate a single palm trigger point based on finger positions and spread/openness.
        
        The trigger point moves based on finger spread:
        - When fingers are closed/fisted: point is near palm center (wrist + MCP joints)
        - When fingers are spread: point moves towards fingertip average
        - The movement is proportional to finger spread
        
        Args:
            landmarks: List of 21 (x, y, z) tuples from MediaPipe hand landmarks
            
        Returns:
            (x, y, z) tuple of the trigger point in normalized coordinates, or None if invalid
        """
        if len(landmarks) < 21:
            return None
        
        # MediaPipe hand landmark indices
        WRIST = 0
        THUMB_TIP = 4
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20
        INDEX_MCP = 5
        MIDDLE_MCP = 9
        RING_MCP = 13
        PINKY_MCP = 17
        
        # Calculate palm base center (wrist + MCP joints)
        wrist = landmarks[WRIST]
        index_mcp = landmarks[INDEX_MCP]
        middle_mcp = landmarks[MIDDLE_MCP]
        ring_mcp = landmarks[RING_MCP]
        pinky_mcp = landmarks[PINKY_MCP]
        
        palm_base = (
            (wrist[0] + index_mcp[0] + middle_mcp[0] + ring_mcp[0] + pinky_mcp[0]) / 5.0,
            (wrist[1] + index_mcp[1] + middle_mcp[1] + ring_mcp[1] + pinky_mcp[1]) / 5.0,
            (wrist[2] + index_mcp[2] + middle_mcp[2] + ring_mcp[2] + pinky_mcp[2]) / 5.0
        )
        
        # Calculate fingertip average (excluding thumb for spread calculation)
        index_tip = landmarks[INDEX_TIP]
        middle_tip = landmarks[MIDDLE_TIP]
        ring_tip = landmarks[RING_TIP]
        pinky_tip = landmarks[PINKY_TIP]
        
        fingertips_avg = (
            (index_tip[0] + middle_tip[0] + ring_tip[0] + pinky_tip[0]) / 4.0,
            (index_tip[1] + middle_tip[1] + ring_tip[1] + pinky_tip[1]) / 4.0,
            (index_tip[2] + middle_tip[2] + ring_tip[2] + pinky_tip[2]) / 4.0
        )
        
        # Calculate finger spread metric
        # Measure distances from each fingertip to the palm base
        spread_distances = []
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
            dist = np.sqrt(
                (tip[0] - palm_base[0]) ** 2 +
                (tip[1] - palm_base[1]) ** 2 +
                (tip[2] - palm_base[2]) ** 2
            )
            spread_distances.append(dist)
        
        # Average spread distance
        avg_spread = np.mean(spread_distances)
        
        # Normalize spread (typical range: 0.05 to 0.25 in normalized space)
        # When fingers are closed: spread ~ 0.05-0.08
        # When fingers are fully spread: spread ~ 0.15-0.25
        min_spread = 0.05
        max_spread = 0.25
        normalized_spread = np.clip((avg_spread - min_spread) / (max_spread - min_spread), 0.0, 1.0)
        
        # Interpolate between palm base and fingertip average based on spread
        # When spread = 0 (closed): use palm_base (weight = 1.0)
        # When spread = 1 (open): use fingertips_avg (weight = 1.0)
        # The trigger point moves more towards fingertips when fingers are spread open
        # Use a smooth curve to make it more responsive - square the normalized spread for smoother transition
        spread_curve = normalized_spread ** 0.7  # Slight curve for smoother transition
        spread_factor = spread_curve * 0.95  # Move up to 95% towards fingertips when fully spread
        
        trigger_point = (
            palm_base[0] * (1.0 - spread_factor) + fingertips_avg[0] * spread_factor,
            palm_base[1] * (1.0 - spread_factor) + fingertips_avg[1] * spread_factor,
            palm_base[2] * (1.0 - spread_factor) + fingertips_avg[2] * spread_factor
        )
        
        return trigger_point
    
    def update_hit_tracking(self, hand_landmarks_list: List[dict], frame_index: Optional[int] = None):
        """
        Update voxel hit tracking based on a single palm trigger point per hand.
        
        The trigger point is calculated based on finger positions and spread/openness.
        Only this single point triggers grid voxels, not individual landmarks.
        
        Args:
            hand_landmarks_list: List of dicts with 'landmarks' key containing
                                list of (x, y, z) tuples (z from MediaPipe)
            frame_index: Optional frame index for timestamp tracking
        """
        if self.voxel_hit_bool is None:
            self.reset_hit_tracking()
        
        if self.voxel_centers is None:
            return
        
        # Process each hand - use only the single trigger point
        for hand_data in hand_landmarks_list:
            landmarks = hand_data.get('landmarks', [])
            
            if len(landmarks) < 21:
                continue
            
            # Calculate single palm trigger point based on finger spread
            trigger_point = self.calculate_palm_trigger_point(landmarks)
            
            if trigger_point is None:
                self.current_trigger_point_2d = None
                self.current_trigger_point_3d = None
                continue
            
            lx_norm, ly_norm, lz_mp = trigger_point
            
            # Store trigger point for visualization
            self.current_trigger_point_3d = trigger_point
            if self.frame_width and self.frame_height:
                pixel_u = int(lx_norm * self.frame_width)
                pixel_v = int(ly_norm * self.frame_height)
                self.current_trigger_point_2d = (pixel_u, pixel_v)
            else:
                self.current_trigger_point_2d = None
            
            # Dynamic Depth Scaling: Transform raw Z to relative Z
            # Relative_Z = (Raw_Z - Face_Z_Reference) / Reference_Length
            relative_z = self.calculate_relative_z(lz_mp)
            
            # Determine layer based on dynamic depth scaling (relative_z)
            if relative_z is not None:
                # Dynamic Layer Switching:
                # Layer 0 (Face Layer): relative_z >= 0 (hand at or behind face)
                # Layer 1 (Forward Layer): relative_z < 0 (hand extended toward camera)
                if relative_z < 0:
                    matching_layer_idx = 1  # Forward Layer (near camera)
                else:
                    matching_layer_idx = 0  # Face Layer (near face)
            else:
                # No valid reference data yet; skip this hand
                continue
            
            # Select voxels from the matching layer only
            voxels_per_layer = self.breadth * self.length
            layer_start_idx = matching_layer_idx * voxels_per_layer
            layer_end_idx = layer_start_idx + voxels_per_layer
            
            # Debug output
            layer_name = "Face Layer" if matching_layer_idx == 0 else "Forward Layer"
            print(f"[Layer {matching_layer_idx} ({layer_name})] Relative_Z={relative_z:.3f} → Voxels {layer_start_idx}-{layer_end_idx-1}")
            
            # Find nearest voxel within the matching layer (check x, y only since z already matched)
            min_dist_2d = float('inf')
            nearest_voxel_idx = -1
            
            for voxel_idx in range(layer_start_idx, layer_end_idx):
                vx, vy, vz = self.voxel_centers[voxel_idx]
                
                # Calculate 2D distance (x, y only) since z already matched
                dist_2d = np.sqrt(
                    (lx_norm - vx) ** 2 +
                    (ly_norm - vy) ** 2
                )
                
                if dist_2d < min_dist_2d:
                    min_dist_2d = dist_2d
                    nearest_voxel_idx = voxel_idx
            
            # Use 2D hit radius since z already matched
            hit_radius_2d = self.hit_radius_norm
            
            if nearest_voxel_idx >= 0 and min_dist_2d <= hit_radius_2d:
                # Only append if this is a different voxel than the last one
                # This prevents duplicate consecutive entries in the hit order
                if len(self.voxel_hit_order) == 0 or self.voxel_hit_order[-1] != nearest_voxel_idx:
                    if not self.voxel_hit_bool[nearest_voxel_idx]:
                        self.voxel_hit_bool[nearest_voxel_idx] = True
                    self.voxel_hit_order.append(nearest_voxel_idx)
                    
                    # Calculate chain code direction for trajectory validation
                    self._update_chain_code(nearest_voxel_idx)
    
    def draw_grid(
        self,
        frame: np.ndarray,
        show_hits: bool = True,
        selected_landmark: Optional[int] = None,
        show_indices: bool = True  # Always show indices by default
    ) -> np.ndarray:
        """
        Draw the 3D voxel grid on the frame with proper 3D layered rendering (back-to-front).
        
        Args:
            frame: BGR frame to draw on
            show_hits: If True, color hit voxels differently (green)
            selected_landmark: Optional landmark index to highlight path for
            show_indices: If True, overlay voxel indices for numbering reference
            
        Returns:
            Frame with grid drawn
        """
        if self.voxel_centers_2d is None or self.voxel_centers is None:
            return frame
        
        annotated_frame = frame.copy()
        
        # Create list of voxels with their z-depth for sorting (back-to-front)
        # Apply a small horizontal offset per layer to make layers visually distinguishable
        layer_offset_pixels = 3  # Pixels to offset each layer horizontally
        
        voxel_draw_list = []
        for voxel_idx in range(self.num_voxels):
            if voxel_idx < len(self.voxel_centers) and voxel_idx < len(self.voxel_centers_2d):
                u, v = self.voxel_centers_2d[voxel_idx]
                _, _, z_norm = self.voxel_centers[voxel_idx]
                z_idx = voxel_idx // (self.breadth * self.length)
                
                # Apply horizontal offset based on layer (farther layers offset more to the right)
                u_offset = u + z_idx * layer_offset_pixels
                
                if 0 <= u_offset < self.frame_width and 0 <= v < self.frame_height:
                    voxel_draw_list.append((voxel_idx, u_offset, v, z_norm, z_idx))
        
        # Sort by z-depth (farthest first, so nearer voxels overlay)
        voxel_draw_list.sort(key=lambda x: x[3], reverse=True)
        
        # Colors for different depth layers (near to far) - more distinguishable
        layer_colors = [
            np.array((0, 255, 0)),      # Near layer - bright green
            np.array((0, 255, 255)),    # Middle layer - bright yellow
            np.array((255, 0, 255)),    # Far layer - bright magenta
        ]
        # Extend colors if more than 3 layers
        while len(layer_colors) < self.depth_layers:
            layer_colors.append(np.array((255, 255, 0)))  # Default bright cyan for extra layers

        # Draw voxels back-to-front
        for voxel_idx, u, v, z_norm, z_idx in voxel_draw_list:
            # Determine color and size based on hit status and depth layer
            is_hit = self.voxel_hit_bool is not None and self.voxel_hit_bool[voxel_idx]
            
            # Depth factor: nearer layers (z_idx=0) are larger and brighter
            layer_depth_factor = 1.0 - (z_idx / max(self.depth_layers - 1, 1))
            base_color = layer_colors[min(z_idx, len(layer_colors) - 1)]
            
            if show_hits and is_hit:
                # Layer 1 (z_idx=1, near camera) should be bright red when triggered
                if z_idx == 1:  # Layer 1 (near camera, points 80-159)
                    color = (0, 0, 255)  # Bright red in BGR format
                    brightness = 1.0
                    base_radius = 8  # Larger radius for Layer 1 to make it more visible
                    radius = max(6, int(base_radius * (0.9 + layer_depth_factor * 0.1)))
                else:
                    # Bright color for hit voxels in other layers, larger for nearer layers
                    brightness = 0.8 + layer_depth_factor * 0.2
                    color_vec = np.clip(base_color * brightness / 255.0, 0, 1)
                    color = tuple(int(c * 255) for c in color_vec)
                    base_radius = 6
                    radius = max(4, int(base_radius * (0.8 + layer_depth_factor * 0.4)))
            else:
                # Dimmer color for unhit voxels, smaller for farther layers
                brightness = 0.4 + layer_depth_factor * 0.3
                color_vec = base_color * brightness / 255.0
                color = tuple(int(c * 255) for c in np.clip(color_vec, 0, 1))
                base_radius = 4
                radius = max(3, int(base_radius * (0.6 + layer_depth_factor * 0.4)))
            
            # Draw circle for voxel
            cv2.circle(annotated_frame, (u, v), radius, color, -1)

            if show_indices:
                # Enhanced number visibility with better contrast
                label_color = (255, 255, 255)  # Always white for better visibility
                # Add a small black outline for better contrast
                cv2.putText(
                    annotated_frame,
                    str(voxel_idx),
                    (u + 2, v - 2),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.8,  # Slightly larger font
                    (0, 0, 0),  # Black outline
                    2,  # Thicker outline
                    cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame,
                    str(voxel_idx),
                    (u + 2, v - 2),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.8,  # Slightly larger font
                    label_color,
                    1,
                    cv2.LINE_AA
                )
        
        # Draw path line showing the order of hits (from voxel_hit_order)
        # This draws a line connecting all hit voxels in the order they were hit
        if self.voxel_hit_order and len(self.voxel_hit_order) > 1:
            path_points = []
            for voxel_idx in self.voxel_hit_order:
                if voxel_idx < len(self.voxel_centers_2d):
                    u, v = self.voxel_centers_2d[voxel_idx]
                    # Apply layer offset for consistency
                    z_idx = voxel_idx // (self.breadth * self.length)
                    layer_offset_pixels = 3
                    u_offset = u + z_idx * layer_offset_pixels
                    if 0 <= u_offset < self.frame_width and 0 <= v < self.frame_height:
                        path_points.append((u_offset, v))
            
            if len(path_points) > 1:
                pts = np.array(path_points, np.int32)
                # Draw a bright magenta line to show the path (reduced thickness)
                cv2.polylines(annotated_frame, [pts], False, (255, 0, 255), 1)
                # Also draw small circles at each point in the path for visibility (reduced size)
                for pt in path_points:
                    cv2.circle(annotated_frame, pt, 2, (255, 0, 255), -1)
        
        # Draw trigger point (palm center based on finger spread)
        if self.current_trigger_point_2d is not None:
            u, v = self.current_trigger_point_2d
            if 0 <= u < self.frame_width and 0 <= v < self.frame_height:
                # Draw a large, bright red circle to indicate the trigger point
                cv2.circle(annotated_frame, (u, v), 8, (0, 0, 255), -1)  # Red filled circle
                cv2.circle(annotated_frame, (u, v), 10, (255, 255, 255), 2)  # White outline
                # Label it
                cv2.putText(
                    annotated_frame,
                    "TRIGGER",
                    (u + 12, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
        
        # Draw status text
        hit_count = np.sum(self.voxel_hit_bool) if self.voxel_hit_bool is not None else 0
        status_text = f"Depth: MediaPipe z (relative)"
        status_text += f" | Hit Voxels: {hit_count} / {self.num_voxels}"
        status_text += f" | Head Yaw: {self.head_yaw_deg:.1f}°"
        status_text += f" | Head Pitch: {self.head_pitch_deg:.1f}°"
        if selected_landmark is not None:
            landmark_name = LANDMARK_NAMES.get(selected_landmark, f"landmark_{selected_landmark}")
            status_text += f" | Selected Landmark: {landmark_name}"
        
        label_color = (0, 165, 255)
        cv2.putText(annotated_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
        
        # Display trigger z value at all times
        if self.current_trigger_point_3d is not None:
            trigger_z_raw = self.current_trigger_point_3d[2]  # Raw MediaPipe z
            trigger_z_normalized = np.clip((trigger_z_raw + 0.5) / 1.0, 0.0, 1.0)
            if trigger_z_normalized < 0.01:
                trigger_z_normalized = 0.5
            trigger_z_text = f"Trigger Z: {trigger_z_normalized:.3f}"
            cv2.putText(annotated_frame, trigger_z_text, (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
        
        # Draw chain code debug information (real-time signature generation)
        chain_code_str = ','.join([str(d) for d in self.current_gesture_chain]) if self.current_gesture_chain else "[]"
        chain_code_display = f"Chain Code: [{chain_code_str}]"
        
        # Draw chain code text at bottom of frame (cyan color for visibility)
        cv2.putText(
            annotated_frame,
            chain_code_display,
            (10, self.frame_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated_frame,
            chain_code_display,
            (10, self.frame_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return annotated_frame
    
    def release(self):
        """Release MediaPipe Face Mesh resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()
