"""
Face-centered 3D grid using MiDaS depth estimation.

This module constructs a 3D grid centered on the face using MediaPipe FaceMesh
for landmark detection and MiDaS for depth estimation.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple
from midas_depth import MiDaSDepthProvider


class FaceGridMiDaS:
    """
    Tracks a face-centered 3D grid using MiDaS depth estimation.
    
    The grid is constructed around the nose center point, with spacing based on
    the distance between the nose and eye centers. The z-coordinate (depth) is
    derived from MiDaS depth estimation rather than MediaPipe's relative depth.
    """
    
    # MediaPipe Face Mesh landmark indices
    NOSE_TIP = 4  # Nose tip (stable point)
    LEFT_EYE_INNER = 133  # Left eye inner corner
    LEFT_EYE_OUTER = 33  # Left eye outer corner
    RIGHT_EYE_INNER = 362  # Right eye inner corner
    RIGHT_EYE_OUTER = 263  # Right eye outer corner
    
    def __init__(self, grid_width: int = 12, grid_height: int = 15, depth_provider: Optional[MiDaSDepthProvider] = None):
        """
        Initialize the face grid tracker with MiDaS depth.
        
        Args:
            grid_width: Number of grid points horizontally (x-axis, default: 12)
            grid_height: Number of grid points vertically (y-axis, default: 15)
            depth_provider: Optional MiDaSDepthProvider instance. If None, will create one.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_grid_points = grid_width * grid_height
        
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
        self.grid_points_3d = None  # List of (x, y, z) tuples in normalized coordinates
        self.grid_points_2d = None  # List of (x, y) tuples in pixel coordinates
        self.hit_grid = None  # 2D boolean array tracking which points were hit [height, width]
        self.frame_width = None
        self.frame_height = None
        self.grid_center_z = 0.0  # MiDaS depth at nose pixel
        self.grid_spacing_norm = 0.0  # Normalized spacing used
        
    def initialize_depth(self) -> bool:
        """
        Initialize the MiDaS depth provider.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self.depth_provider.is_active():
            return self.depth_provider.start()
        return True
    
    def reset_hit_grid(self):
        """Reset the hit grid tracking (call at start of new capture)."""
        self.hit_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
    
    def get_hit_grid_vector(self) -> List[int]:
        """
        Get the hit grid as a flattened 1D vector of 0s and 1s.
        
        Returns:
            List of integers (0 or 1) of length grid_width * grid_height
            Order: row by row (left to right, top to bottom)
        """
        if self.hit_grid is None:
            return [0] * self.num_grid_points
        
        # Flatten: row by row (left to right, top to bottom)
        return [1 if self.hit_grid[gy, gx] else 0 
                for gy in range(self.grid_height) 
                for gx in range(self.grid_width)]
    
    def get_hit_points_coordinates(self) -> List[Tuple[float, float, float]]:
        """
        Get the 3D coordinates of all hit grid points.
        
        Returns:
            List of (x, y, z) tuples for each grid point that was hit
        """
        if self.hit_grid is None or self.grid_points_3d is None:
            return []
        
        hit_points = []
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if self.hit_grid[gy, gx]:
                    # Get the 3D coordinates of this grid point
                    idx = gy * self.grid_width + gx
                    hit_points.append(self.grid_points_3d[idx])
        
        return hit_points
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Process a frame to detect face and build/update the grid using MiDaS depth.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            True if face detected and grid built, False otherwise
        """
        if self.face_mesh is None:
            return False
        
        # Store frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Process depth map first
        depth_available = False
        if self.depth_provider.is_active():
            depth_available = self.depth_provider.process_frame(frame)
        
        if not depth_available:
            # If depth fails, we can still build grid but with default depth
            print("Warning: MiDaS depth not available, using default depth")
        
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
        
        # Calculate eye centers (average of inner and outer corners)
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
        
        # Get MiDaS depth at nose pixel
        nose_u = int(nose_center[0] * self.frame_width)
        nose_v = int(nose_center[1] * self.frame_height)
        nose_depth = self.depth_provider.get_depth_at(nose_u, nose_v) if depth_available else 0.5
        self.grid_center_z = nose_depth
        
        # Get MiDaS depth at eye pixels
        left_eye_u = int(left_eye_center[0] * self.frame_width)
        left_eye_v = int(left_eye_center[1] * self.frame_height)
        left_eye_depth = self.depth_provider.get_depth_at(left_eye_u, left_eye_v) if depth_available else nose_depth
        
        # Calculate horizontal spacing unit (distance from nose to left eye in pixel coords)
        dx_pixels = np.sqrt(
            (nose_center[0] - left_eye_center[0]) ** 2 * self.frame_width ** 2 +
            (nose_center[1] - left_eye_center[1]) ** 2 * self.frame_height ** 2
        )
        
        # Convert to normalized coordinates
        dx_norm = np.sqrt(
            (nose_center[0] - left_eye_center[0]) ** 2 +
            (nose_center[1] - left_eye_center[1]) ** 2
        )
        
        # Scale by relative depth to keep grid stable
        # If depth is available, adjust spacing based on depth
        if depth_available and nose_depth > 0:
            # Deeper objects should have larger spacing (inverse relationship)
            depth_scale = 1.0 / (nose_depth + 0.1)  # Add small epsilon to avoid division issues
            dx_norm = dx_norm * depth_scale
        
        # If distance is too small, use a default
        if dx_norm < 1e-6:
            dx_norm = 0.05  # Default spacing
        
        # Calculate vertical spacing (proportional to horizontal)
        eye_distance = np.sqrt(
            (left_eye_center[0] - right_eye_center[0]) ** 2 +
            (left_eye_center[1] - right_eye_center[1]) ** 2
        )
        dy_norm = eye_distance * 1.2  # Slightly larger for vertical spacing
        
        self.grid_spacing_norm = (dx_norm + dy_norm) / 2  # Average spacing
        
        # Build the 2D grid (12 columns × 15 rows)
        # Grid extends around the nose center
        grid_points_3d = []
        grid_points_2d = []
        
        # Calculate grid bounds
        half_width = (self.grid_width - 1) / 2
        half_height = (self.grid_height - 1) / 2
        
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                # Calculate offset from center in normalized coordinates
                offset_x = (gx - half_width) * dx_norm
                offset_y = (gy - half_height) * dy_norm
                
                # Create grid point in 2D normalized space
                grid_x = nose_center[0] + offset_x
                grid_y = nose_center[1] + offset_y
                
                # Clamp to valid range [0, 1]
                grid_x = np.clip(grid_x, 0.0, 1.0)
                grid_y = np.clip(grid_y, 0.0, 1.0)
                
                # Get MiDaS depth at this grid point
                grid_u = int(grid_x * self.frame_width)
                grid_v = int(grid_y * self.frame_height)
                grid_z = self.depth_provider.get_depth_at(grid_u, grid_v) if depth_available else nose_depth
                
                grid_points_3d.append((grid_x, grid_y, grid_z))
                
                # Convert to pixel coordinates for drawing
                pixel_x = int(grid_x * self.frame_width)
                pixel_y = int(grid_y * self.frame_height)
                grid_points_2d.append((pixel_x, pixel_y))
        
        self.grid_points_3d = grid_points_3d
        self.grid_points_2d = grid_points_2d
        
        return True
    
    def update_hit_grid(self, hand_landmarks_list: List[dict], hit_radius: float = 0.03):
        """
        Update the hit grid based on hand landmarks using MiDaS depth.
        
        For each hand landmark, find the closest grid point in 3D space (x, y, z)
        where z comes from MiDaS depth, and mark it as hit if within the hit radius.
        
        Args:
            hand_landmarks_list: List of dicts with 'landmarks' key containing
                                list of (x, y, z) tuples (z from MediaPipe, but we use MiDaS)
            hit_radius: Distance threshold in normalized coordinates (default: 0.03)
        """
        if self.hit_grid is None:
            self.reset_hit_grid()
        
        if self.grid_points_3d is None:
            return
        
        # Get current depth map
        depth_map = self.depth_provider.get_depth_map()
        if depth_map is None:
            return
        
        # Process all hand landmarks
        for hand_data in hand_landmarks_list:
            landmarks = hand_data.get('landmarks', [])
            
            for landmark in landmarks:
                lx_norm, ly_norm, lz_mp = landmark  # MediaPipe z (we'll replace with MiDaS)
                
                # Convert normalized landmark to pixel coordinates
                lu = int(lx_norm * self.frame_width)
                lv = int(ly_norm * self.frame_height)
                
                # Get MiDaS depth at hand landmark location
                lz_midas = self.depth_provider.get_depth_at(lu, lv)
                
                # Construct 3D point using MiDaS depth
                hand_point_3d = (lx_norm, ly_norm, lz_midas)
                
                # Find closest grid point in 3D space
                min_dist = float('inf')
                closest_gx = 0
                closest_gy = 0
                
                for idx, (gx, gy, gz) in enumerate(self.grid_points_3d):
                    # Calculate 3D Euclidean distance in normalized+depth space
                    dist = np.sqrt(
                        (lx_norm - gx) ** 2 +
                        (ly_norm - gy) ** 2 +
                        (lz_midas - gz) ** 2
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        # Convert linear index to 2D indices
                        closest_gy = idx // self.grid_width
                        closest_gx = idx % self.grid_width
                
                # Mark as hit if within radius
                if min_dist <= hit_radius:
                    self.hit_grid[closest_gy, closest_gx] = True
    
    def draw_grid(self, frame: np.ndarray, show_hits: bool = True, depth_mode: str = "midas") -> np.ndarray:
        """
        Draw the grid on the frame with hit visualization.
        
        Args:
            frame: BGR frame to draw on
            show_hits: If True, color hit points differently (green)
            depth_mode: Current depth mode ("midas" or "off")
            
        Returns:
            Frame with grid drawn
        """
        if self.grid_points_2d is None:
            return frame
        
        annotated_frame = frame.copy()
        
        # Draw grid points
        for idx, (px, py) in enumerate(self.grid_points_2d):
            # Check if point is within frame bounds
            if 0 <= px < self.frame_width and 0 <= py < self.frame_height:
                # Convert linear index to 2D indices
                gy = idx // self.grid_width
                gx = idx % self.grid_width
                
                # Determine color based on hit status
                if show_hits and self.hit_grid is not None and self.hit_grid[gy, gx]:
                    # Green for hit points
                    color = (0, 255, 0)
                    radius = 4
                else:
                    # Gray for unhit points
                    color = (128, 128, 128)
                    radius = 2
                
                # Draw circle for grid point
                cv2.circle(annotated_frame, (px, py), radius, color, -1)
        
        # Draw depth mode label
        label_text = f"Depth Mode: {depth_mode.upper()}" if depth_mode == "midas" else "Depth Mode: OFFLINE"
        label_color = (0, 255, 0) if depth_mode == "midas" else (0, 0, 255)
        cv2.putText(annotated_frame, label_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
        
        return annotated_frame
    
    def release(self):
        """Release MediaPipe Face Mesh and depth provider resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.own_depth_provider and self.depth_provider is not None:
            self.depth_provider.release()

