"""
Face-centered 3D voxel grid using MiDaS depth estimation.

This module constructs a 3D voxel cube (rectangular prism) centered on the face
using MediaPipe FaceMesh for landmark detection and MiDaS for depth estimation.
Records multi-layer voxel hits and ordered voxel travel paths per landmark.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict
from midas_depth import MiDaSDepthProvider


# MediaPipe Hand landmark indices (for path tracking)
INDEX_TIP = 8
THUMB_TIP = 4
LANDMARK_NAMES = {
    0: "wrist",
    1: "thumb_cmc", 2: "thumb_mcp", 3: "thumb_ip", 4: "thumb_tip",
    5: "index_mcp", 6: "index_pip", 7: "index_dip", 8: "index_tip",
    9: "middle_mcp", 10: "middle_pip", 11: "middle_dip", 12: "middle_tip",
    13: "ring_mcp", 14: "ring_pip", 15: "ring_dip", 16: "ring_tip",
    17: "pinky_mcp", 18: "pinky_pip", 19: "pinky_dip", 20: "pinky_tip"
}


class FaceGrid3D:
    """
    Tracks a face-centered 3D voxel grid using MiDaS depth estimation or MediaPipe z fallback.
    
    The grid is a rectangular prism (voxel cube) centered on the nose center point,
    with spacing based on the distance between the nose and eye centers.
    The z-coordinate (depth) is derived from MiDaS depth estimation when available,
    otherwise falls back to MediaPipe landmark z values (normalized to 0-1 range).
    
    Voxel indexing order: [z, row, col] - z (depth layer) is fastest, then row (y), then col (x)
    This means: idx = z * (breadth * length) + row * breadth + col
    
    Ordering for hit_grid_3d and voxel_paths:
    - Flattened array order: z-major (layer by layer from near→far)
    - Within each layer: row-major over Y then X
    - Example for grid_dims=[12,15,7]: 
      idx = z_idx * (12 * 15) + y_idx * 12 + x_idx
      where z_idx ∈ [0,6], y_idx ∈ [0,14], x_idx ∈ [0,11]
    """
    
    # MediaPipe Face Mesh landmark indices
    NOSE_TIP = 4  # Nose tip (stable point)
    LEFT_EYE_INNER = 133  # Left eye inner corner
    LEFT_EYE_OUTER = 33  # Left eye outer corner
    RIGHT_EYE_INNER = 362  # Right eye inner corner
    RIGHT_EYE_OUTER = 263  # Right eye outer corner
    
    def __init__(
        self,
        breadth: int = 12,  # x-axis (width)
        length: int = 15,   # y-axis (height)
        depth_layers: int = 7,  # z-axis (depth)
        depth_provider: Optional[MiDaSDepthProvider] = None,
        depth_span_factor: float = 1.5,  # How many spacing steps span forward/back
        hit_radius_norm: float = 0.035,  # Hit detection radius in normalized space
        track_landmark_paths: bool = True,
        tracked_landmarks: Optional[List[int]] = None
    ):
        """
        Initialize the 3D voxel grid tracker with MiDaS depth.
        
        Args:
            breadth: Number of voxels horizontally (x-axis, default: 12)
            length: Number of voxels vertically (y-axis, default: 15)
            depth_layers: Number of depth layers (z-axis, default: 7)
            depth_provider: Optional MiDaSDepthProvider instance. If None, will create one.
            depth_span_factor: Multiplier for depth layer span (default: 1.5)
            hit_radius_norm: Hit detection radius in normalized coordinates (default: 0.035)
            track_landmark_paths: If True, record ordered paths per landmark (default: True)
            tracked_landmarks: List of landmark indices to track paths for (default: [8, 4] = index_tip, thumb_tip)
        """
        self.breadth = breadth
        self.length = length
        self.depth_layers = depth_layers
        self.num_voxels = breadth * length * depth_layers  # 12 * 15 * 7 = 1260
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
        
        # MiDaS depth provider
        if depth_provider is None:
            self.depth_provider = MiDaSDepthProvider(model_type="DPT_Large")
            self.own_depth_provider = True
        else:
            self.depth_provider = depth_provider
            self.own_depth_provider = False
        
        # Grid state
        self.voxel_centers = None  # List of (x, y, z) tuples in normalized coordinates
        self.voxel_centers_2d = None  # List of (u, v) tuples in pixel coordinates (for visualization)
        self.voxel_hit_bool = None  # Flattened boolean array [num_voxels]
        self.voxel_paths = None  # Dict mapping landmark_idx -> list of voxel indices (ordered)
        self.voxel_last_index_per_landmark = None  # Dict mapping landmark_idx -> last voxel index
        self.frame_width = None
        self.frame_height = None
        self.grid_center_z = 0.0  # Depth at nose pixel (MiDaS or MediaPipe z)
        self.grid_spacing_norm = 0.0  # Normalized spacing used
        self.depth_mode = "midas"  # Current depth mode: "midas" or "off" (MediaPipe fallback)
        
        # Temporal smoothing for depth (simple moving average)
        self.depth_history = []
        self.depth_history_size = 3
        
        # MiDaS stride support (process depth every N frames)
        self.midas_stride = 1  # Process every frame by default
        self.midas_frame_counter = 0
        self.last_depth_map = None
    
    def initialize_depth(self) -> bool:
        """
        Initialize the MiDaS depth provider.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self.depth_provider.is_active():
            return self.depth_provider.start()
        return True
    
    def reset_hit_tracking(self):
        """Reset hit tracking and paths (call at start of new capture)."""
        self.voxel_hit_bool = np.zeros(self.num_voxels, dtype=bool)
        self.voxel_paths = {landmark_idx: [] for landmark_idx in self.tracked_landmarks}
        self.voxel_last_index_per_landmark = {landmark_idx: -1 for landmark_idx in self.tracked_landmarks}
        self.depth_history = []
    
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
    
    def _get_smoothed_depth(self, depth: float) -> float:
        """Apply temporal smoothing to depth values."""
        self.depth_history.append(depth)
        if len(self.depth_history) > self.depth_history_size:
            self.depth_history.pop(0)
        return np.mean(self.depth_history) if self.depth_history else depth
    
    def set_depth_mode(self, depth_mode: str):
        """
        Set depth mode: "midas" or "off" (MediaPipe z fallback).
        
        Args:
            depth_mode: "midas" to use MiDaS depth, "off" to use MediaPipe z
        """
        self.depth_mode = depth_mode
    
    def set_midas_stride(self, stride: int):
        """
        Set MiDaS processing stride (process depth every N frames).
        
        Args:
            stride: Process depth every N frames (default: 1 = every frame)
        """
        self.midas_stride = max(1, stride)
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Process a frame to detect face and build/update the 3D voxel grid.
        Uses MiDaS depth if available, otherwise falls back to MediaPipe z.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            True if face detected and grid built, False otherwise
        """
        if self.face_mesh is None:
            return False
        
        # Store frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Process depth map (with stride support)
        depth_available = False
        use_midas_depth = False
        
        if self.depth_mode == "midas" and self.depth_provider.is_active():
            # Check if we should process depth this frame
            self.midas_frame_counter += 1
            if self.midas_frame_counter >= self.midas_stride:
                depth_available = self.depth_provider.process_frame(frame)
                self.midas_frame_counter = 0
                if depth_available:
                    self.last_depth_map = self.depth_provider.get_depth_map()
                    use_midas_depth = True
            elif self.last_depth_map is not None:
                # Reuse last depth map
                depth_available = True
                use_midas_depth = True
        
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
        
        # Nose center (using nose tip)
        nose_center = (nose_tip.x, nose_tip.y)
        nose_center_z_mp = nose_tip.z  # MediaPipe z (relative depth)
        
        # Get depth at nose pixel (MiDaS or MediaPipe fallback)
        nose_u = int(nose_center[0] * self.frame_width)
        nose_v = int(nose_center[1] * self.frame_height)
        
        if use_midas_depth and self.depth_provider.is_active():
            # Use MiDaS depth with neighborhood sampling for stability
            neighborhood_size = 3
            depth_samples = []
            for du in range(-neighborhood_size, neighborhood_size + 1):
                for dv in range(-neighborhood_size, neighborhood_size + 1):
                    u = np.clip(nose_u + du, 0, self.frame_width - 1)
                    v = np.clip(nose_v + dv, 0, self.frame_height - 1)
                    d = self.depth_provider.get_depth_at(u, v)
                    if d > 0:
                        depth_samples.append(d)
            
            if depth_samples:
                nose_depth = np.median(depth_samples)  # Use median for robustness
            else:
                nose_depth = self.depth_provider.get_depth_at(nose_u, nose_v) or 0.5
            
            # Apply temporal smoothing
            nose_depth = self._get_smoothed_depth(nose_depth)
        else:
            # Fallback to MediaPipe z (normalize to 0-1 range)
            # MediaPipe z is relative and can be negative, so we normalize it
            # Map typical MediaPipe z range (-0.5 to 0.5) to 0-1
            nose_depth = np.clip((nose_center_z_mp + 0.5) / 1.0, 0.0, 1.0)
            if nose_depth < 0.01:  # If too close to 0, use a default
                nose_depth = 0.5
        
        self.grid_center_z = nose_depth
        
        # Get depth at eye pixels (for spacing calculation, not critical)
        # We'll use nose_depth as reference
        
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
        dz_norm = dx_norm * self.depth_span_factor / self.depth_layers
        
        # Build the 3D voxel grid
        voxel_centers = []
        voxel_centers_2d = []
        
        half_breadth = (self.breadth - 1) / 2
        half_length = (self.length - 1) / 2
        half_depth = (self.depth_layers - 1) / 2
        
        # Order: [z, row, col] - z fastest, then row, then col
        for z_idx in range(self.depth_layers):
            for y_idx in range(self.length):
                for x_idx in range(self.breadth):
                    # Calculate offset from center
                    offset_x = (x_idx - half_breadth) * dx_norm
                    offset_y = (y_idx - half_length) * dy_norm
                    offset_z = (z_idx - half_depth) * dz_norm
                    
                    # Create voxel center in normalized space
                    voxel_x = nose_center[0] + offset_x
                    voxel_y = nose_center[1] + offset_y
                    voxel_z = nose_depth + offset_z  # Relative to nose depth
                    
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
        
        For each hand landmark, find the nearest voxel center in 3D space and mark it as hit.
        If path tracking is enabled, record ordered sequences for tracked landmarks.
        Uses MiDaS depth if available, otherwise falls back to MediaPipe z.
        
        Args:
            hand_landmarks_list: List of dicts with 'landmarks' key containing
                                list of (x, y, z) tuples (z from MediaPipe)
            frame_index: Optional frame index for timestamp tracking
        """
        if self.voxel_hit_bool is None:
            self.reset_hit_tracking()
        
        if self.voxel_centers is None:
            return
        
        # Get current depth map (if using MiDaS)
        depth_map = None
        use_midas = False
        if self.depth_mode == "midas" and self.depth_provider.is_active():
            depth_map = self.depth_provider.get_depth_map()
            if depth_map is None and self.last_depth_map is not None:
                depth_map = self.last_depth_map
            use_midas = depth_map is not None
        
        # Process all hand landmarks
        for hand_data in hand_landmarks_list:
            landmarks = hand_data.get('landmarks', [])
            
            for landmark_idx, landmark in enumerate(landmarks):
                lx_norm, ly_norm, lz_mp = landmark  # MediaPipe z (used as fallback)
                
                # Get depth for this landmark (MiDaS or MediaPipe fallback)
                if use_midas:
                    # Convert normalized landmark to pixel coordinates
                    lu = int(lx_norm * self.frame_width)
                    lv = int(ly_norm * self.frame_height)
                    
                    # Get MiDaS depth at hand landmark location (with neighborhood averaging)
                    neighborhood_size = 2
                    depth_samples = []
                    for du in range(-neighborhood_size, neighborhood_size + 1):
                        for dv in range(-neighborhood_size, neighborhood_size + 1):
                            u = np.clip(lu + du, 0, self.frame_width - 1)
                            v = np.clip(lv + dv, 0, self.frame_height - 1)
                            d = self.depth_provider.get_depth_at(u, v)
                            if d > 0:
                                depth_samples.append(d)
                    
                    if depth_samples:
                        lz_depth = np.median(depth_samples)
                    else:
                        lz_depth = self.depth_provider.get_depth_at(lu, lv) or 0.5
                else:
                    # Fallback to MediaPipe z (normalize to 0-1 range)
                    # Map typical MediaPipe z range (-0.5 to 0.5) to 0-1
                    lz_depth = np.clip((lz_mp + 0.5) / 1.0, 0.0, 1.0)
                    if lz_depth < 0.01:
                        lz_depth = 0.5
                
                # Construct 3D point using depth (MiDaS or MediaPipe)
                hand_point_3d = (lx_norm, ly_norm, lz_depth)
                
                # Find nearest voxel center in 3D space
                min_dist = float('inf')
                nearest_voxel_idx = -1
                
                for voxel_idx, (vx, vy, vz) in enumerate(self.voxel_centers):
                    # Calculate 3D Euclidean distance in normalized space
                    dist = np.sqrt(
                        (lx_norm - vx) ** 2 +
                        (ly_norm - vy) ** 2 +
                        (lz_midas - vz) ** 2
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_voxel_idx = voxel_idx
                
                # Mark as hit if within radius
                if min_dist <= self.hit_radius_norm and nearest_voxel_idx >= 0:
                    self.voxel_hit_bool[nearest_voxel_idx] = True
                    
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
        depth_mode: str = "midas",
        selected_landmark: Optional[int] = None
    ) -> np.ndarray:
        """
        Draw the 3D voxel grid on the frame with proper 3D layered rendering (back-to-front).
        
        Args:
            frame: BGR frame to draw on
            show_hits: If True, color hit voxels differently (green)
            depth_mode: Current depth mode ("midas" or "off")
            selected_landmark: Optional landmark index to highlight path for
            
        Returns:
            Frame with grid drawn
        """
        if self.voxel_centers_2d is None or self.voxel_centers is None:
            return frame
        
        annotated_frame = frame.copy()
        
        # Create list of voxels with their z-depth for sorting (back-to-front)
        voxel_draw_list = []
        for voxel_idx in range(self.num_voxels):
            if voxel_idx < len(self.voxel_centers) and voxel_idx < len(self.voxel_centers_2d):
                u, v = self.voxel_centers_2d[voxel_idx]
                if 0 <= u < self.frame_width and 0 <= v < self.frame_height:
                    _, _, z_norm = self.voxel_centers[voxel_idx]
                    z_idx = voxel_idx // (self.breadth * self.length)
                    voxel_draw_list.append((voxel_idx, u, v, z_norm, z_idx))
        
        # Sort by z-depth (farthest first, so nearer voxels overlay)
        voxel_draw_list.sort(key=lambda x: x[3], reverse=True)
        
        # Draw voxels back-to-front
        for voxel_idx, u, v, z_norm, z_idx in voxel_draw_list:
            # Determine color and size based on hit status and depth layer
            is_hit = self.voxel_hit_bool is not None and self.voxel_hit_bool[voxel_idx]
            
            # Depth factor: nearer layers are larger and brighter
            # z_norm ranges from ~0 (far) to ~1 (near), so we use (1 - z_norm) for depth cue
            layer_depth_factor = 1.0 - (z_idx / max(self.depth_layers - 1, 1))
            
            if show_hits and is_hit:
                # Green for hit voxels, brighter and larger for nearer layers
                brightness = 0.7 + layer_depth_factor * 0.3
                color = (0, int(255 * brightness), 0)
                base_radius = 4
                radius = max(2, int(base_radius * (1.0 + layer_depth_factor * 0.5)))
            else:
                # Gray for unhit voxels, darker and smaller for farther layers
                brightness = 0.3 + layer_depth_factor * 0.3
                gray_val = int(128 * brightness)
                color = (gray_val, gray_val, gray_val)
                base_radius = 2
                radius = max(1, int(base_radius * (0.5 + layer_depth_factor * 0.5)))
            
            # Draw circle for voxel
            cv2.circle(annotated_frame, (u, v), radius, color, -1)
        
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
        status_text = f"Depth Mode: {depth_mode.upper()}"
        if depth_mode == "off":
            status_text += " (MediaPipe z fallback)"
        status_text += f" | Hit Voxels: {hit_count} / {self.num_voxels}"
        if selected_landmark is not None:
            landmark_name = LANDMARK_NAMES.get(selected_landmark, f"landmark_{selected_landmark}")
            status_text += f" | Selected Landmark: {landmark_name}"
        
        label_color = (0, 255, 0) if depth_mode == "midas" else (0, 165, 255)  # Orange for fallback
        cv2.putText(annotated_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
        
        # Show depth fallback message if using MediaPipe z
        if depth_mode == "off":
            cv2.putText(annotated_frame, "Depth fallback: MediaPipe z (relative)",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        return annotated_frame
    
    def release(self):
        """Release MediaPipe Face Mesh and depth provider resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.own_depth_provider and self.depth_provider is not None:
            self.depth_provider.release()

