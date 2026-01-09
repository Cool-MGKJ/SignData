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


class FaceGrid3D:
    """
    Tracks a face-centered 3D voxel grid using MediaPipe FaceMesh z-depth.
    
    The grid is a rectangular prism (voxel cube) centered on the nose center point,
    with spacing based on the distance between the nose and eye centers.
    The z-coordinate (depth) is derived from MediaPipe landmark z values
    (normalized to 0-1 range).
    
    Voxel indexing order: [z, row, col] - z (depth layer) is fastest, then row (y), then col (x)
    This means: idx = z * (breadth * length) + row * breadth + col
    
    Ordering for hit_grid_3d and voxel_paths:
    - Flattened array order: z-major (layer by layer from near→far)
    - Within each layer: row-major over Y then X
    - Example for grid_dims=[8,10,3]: 
      idx = z_idx * (8 * 10) + y_idx * 8 + x_idx
      where z_idx ∈ [0,2], y_idx ∈ [0,9], x_idx ∈ [0,7]
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
        hit_radius_norm: float = 0.06,  # Hit detection radius in normalized 3D space
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
        
        # Simple depth mode flag kept for backwards-compatible UI text
        self.depth_mode = "off"
    
    def reset_hit_tracking(self):
        """Reset hit tracking and paths (call at start of new capture)."""
        self.voxel_hit_bool = np.zeros(self.num_voxels, dtype=bool)
        self.voxel_paths = {landmark_idx: [] for landmark_idx in self.tracked_landmarks}
        self.voxel_last_index_per_landmark = {landmark_idx: -1 for landmark_idx in self.tracked_landmarks}
        self.voxel_hit_order = []
    
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
    
    def set_depth_mode(self, depth_mode: str):
        """Kept for backward compatibility; depth mode is now always 'off' (MediaPipe z-only)."""
        self.depth_mode = "off"
    
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
        
        # Nose center (using nose tip)
        nose_center = (nose_tip.x, nose_tip.y)
        nose_center_z_mp = nose_tip.z  # MediaPipe z (relative depth)
        
        # Get depth at nose pixel using MediaPipe z only (normalized)
        # MediaPipe z is relative and can be negative, so we normalize it.
        # Map typical MediaPipe z range (-0.5 to 0.5) to 0-1.
        nose_depth = np.clip((nose_center_z_mp + 0.5) / 1.0, 0.0, 1.0)
        if nose_depth < 0.01:  # If too close to 0, use a default
            nose_depth = 0.5
        
        self.grid_center_z = nose_depth
        
        # Calculate horizontal spacing unit (distance from nose to left eye)
        dx_norm = np.sqrt(
            (nose_center[0] - left_eye_center[0]) ** 2 +
            (nose_center[1] - left_eye_center[1]) ** 2
        )
        
        # Scale by relative depth to keep grid stable
        if nose_depth > 0:
            depth_scale = 1.0 / (nose_depth + 0.1)
            dx_norm = dx_norm * depth_scale
        
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
        vertical_vector_3d = np.array([
            chin.x - forehead.x,
            chin.y - forehead.y,
            chin.z - forehead.z
        ])
        pitch_rad = np.arctan2(vertical_vector_3d[2], vertical_vector_3d[1] + 1e-6)
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
                    
                    # Rotate offsets around the vertical axis (yaw)
                    rotated_x = offset_x * cos_yaw + offset_z * sin_yaw
                    rotated_z = -offset_x * sin_yaw + offset_z * cos_yaw
                    
                    # Rotate around horizontal axis (pitch) after yaw rotation
                    # Note: positive pitch (looking down) should move grid up, negative pitch (looking up) should move grid down
                    rotated_y = offset_y * cos_pitch + rotated_z * sin_pitch
                    rotated_z_pitch = -offset_y * sin_pitch + rotated_z * cos_pitch
                    
                    # Create voxel center in normalized space
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
        
        return True
    
    def update_hit_tracking(self, hand_landmarks_list: List[dict], frame_index: Optional[int] = None):
        """
        Update voxel hit tracking and paths based on hand landmarks.
        
        For each hand landmark, find the nearest voxel center in true 3D space and mark it as hit
        if within the hit radius. The hit detection is purely distance-based in 3D - the voxel
        closest to the hand landmark (considering x, y, and z) will be hit if within range.
        
        Args:
            hand_landmarks_list: List of dicts with 'landmarks' key containing
                                list of (x, y, z) tuples (z from MediaPipe)
            frame_index: Optional frame index for timestamp tracking
        """
        if self.voxel_hit_bool is None:
            self.reset_hit_tracking()
        
        if self.voxel_centers is None:
            return
        
        # Process all hand landmarks
        for hand_data in hand_landmarks_list:
            landmarks = hand_data.get('landmarks', [])
            
            for landmark_idx, landmark in enumerate(landmarks):
                lx_norm, ly_norm, lz_mp = landmark  # MediaPipe z (relative depth)
                
                # Get depth for this landmark using MediaPipe z (normalized 0-1)
                # Map typical MediaPipe z range (-0.5 to 0.5) to 0-1.
                lz_depth = np.clip((lz_mp + 0.5) / 1.0, 0.0, 1.0)
                if lz_depth < 0.01:
                    lz_depth = 0.5
                
                # Find nearest voxel center in true 3D space
                min_dist = float('inf')
                nearest_voxel_idx = -1
                
                for voxel_idx, (vx, vy, vz) in enumerate(self.voxel_centers):
                    # Calculate true 3D Euclidean distance in normalized space
                    # All dimensions (x, y, z) are treated equally
                    dist = np.sqrt(
                        (lx_norm - vx) ** 2 +
                        (ly_norm - vy) ** 2 +
                        (lz_depth - vz) ** 2
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_voxel_idx = voxel_idx
                
                # Hit the nearest voxel if within the 3D hit radius
                # No separate depth tolerance check - pure 3D distance determines hit
                if nearest_voxel_idx >= 0 and min_dist <= self.hit_radius_norm:
                    if not self.voxel_hit_bool[nearest_voxel_idx]:
                        self.voxel_hit_bool[nearest_voxel_idx] = True
                    self.voxel_hit_order.append(nearest_voxel_idx)
                    
                    # Update path tracking for tracked landmarks
                    if self.track_landmark_paths and landmark_idx in self.tracked_landmarks:
                        last_idx = self.voxel_last_index_per_landmark.get(landmark_idx, -1)
                        if nearest_voxel_idx != last_idx:
                            self.voxel_paths[landmark_idx].append(nearest_voxel_idx)
                            self.voxel_last_index_per_landmark[landmark_idx] = nearest_voxel_idx
    
    def draw_grid(
        self,
        frame: np.ndarray,
        show_hits: bool = True,
        selected_landmark: Optional[int] = None,
        show_indices: bool = False
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
        
        # Colors for different depth layers (near to far)
        layer_colors = [
            np.array((0, 255, 0)),      # Near layer - green
            np.array((0, 200, 255)),    # Middle layer - yellow/orange
            np.array((0, 100, 255)),    # Far layer - orange/red
        ]
        # Extend colors if more than 3 layers
        while len(layer_colors) < self.depth_layers:
            layer_colors.append(np.array((100, 100, 255)))  # Default red-ish for extra layers

        # Draw voxels back-to-front
        for voxel_idx, u, v, z_norm, z_idx in voxel_draw_list:
            # Determine color and size based on hit status and depth layer
            is_hit = self.voxel_hit_bool is not None and self.voxel_hit_bool[voxel_idx]
            
            # Depth factor: nearer layers (z_idx=0) are larger and brighter
            layer_depth_factor = 1.0 - (z_idx / max(self.depth_layers - 1, 1))
            base_color = layer_colors[min(z_idx, len(layer_colors) - 1)]
            
            if show_hits and is_hit:
                # Bright color for hit voxels, larger for nearer layers
                brightness = 0.6 + layer_depth_factor * 0.4
                color_vec = np.clip(base_color * brightness / 255.0, 0, 1)
                color = tuple(int(c * 255) for c in color_vec)
                base_radius = 5
                radius = max(3, int(base_radius * (0.7 + layer_depth_factor * 0.5)))
            else:
                # Dimmer color for unhit voxels, smaller for farther layers
                brightness = 0.3 + layer_depth_factor * 0.3
                color_vec = base_color * brightness / 255.0
                color = tuple(int(c * 255) for c in np.clip(color_vec, 0, 1))
                base_radius = 3
                radius = max(2, int(base_radius * (0.5 + layer_depth_factor * 0.5)))
            
            # Draw circle for voxel
            cv2.circle(annotated_frame, (u, v), radius, color, -1)

            if show_indices:
                label_color = (255, 255, 255) if z_idx == 0 else (200, 200, 200)
                cv2.putText(
                    annotated_frame,
                    str(voxel_idx),
                    (u + 2, v - 2),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.6,
                    label_color,
                    1,
                    cv2.LINE_AA
                )
        
        # Draw path for selected landmark
        if selected_landmark is not None and self.voxel_paths is not None:
            path = self.voxel_paths.get(selected_landmark, [])
            if len(path) > 1:
                # Draw polyline connecting voxel centers
                path_points = []
                for voxel_idx in path:
                    if voxel_idx < len(self.voxel_centers_2d):
                        u, v = self.voxel_centers_2d[voxel_idx]
                        if 0 <= u < self.frame_width and 0 <= v < self.frame_height:
                            path_points.append((u, v))
                
                if len(path_points) > 1:
                    pts = np.array(path_points, np.int32)
                    cv2.polylines(annotated_frame, [pts], False, (255, 0, 255), 2)
        
        # Draw status text
        hit_count = np.sum(self.voxel_hit_bool) if self.voxel_hit_bool is not None else 0
        status_text = f"Depth Mode: OFF (MediaPipe z)"
        status_text += f" | Hit Voxels: {hit_count} / {self.num_voxels}"
        status_text += f" | Head Yaw: {self.head_yaw_deg:.1f}°"
        status_text += f" | Head Pitch: {self.head_pitch_deg:.1f}°"
        if selected_landmark is not None:
            landmark_name = LANDMARK_NAMES.get(selected_landmark, f"landmark_{selected_landmark}")
            status_text += f" | Selected Landmark: {landmark_name}"
        
        label_color = (0, 165, 255)
        cv2.putText(annotated_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
        
        # Note: depth is always MediaPipe z (relative)
        cv2.putText(annotated_frame, "Depth: MediaPipe z (relative)",
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        return annotated_frame
    
    def release(self):
        """Release MediaPipe Face Mesh resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()
