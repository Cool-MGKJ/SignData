
"""
Webcam capture and MediaPipe hand landmark detection module.

This module handles:
- Webcam initialization and frame capture
- MediaPipe Hands initialization and processing
- Hand landmark extraction (3D coordinates)
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple
from face_grid_3d import FaceGrid3D


class FaceGridTracker:
    """
    Tracks a face-centered 3D grid and detects which grid points are hit by hand landmarks.
    
    The grid is constructed around the nose center point, with spacing based on
    the distance between the nose and eye centers. The grid is a 3D volume in space,
    with points distributed in width (x), height (y), and depth (z) dimensions.
    """
    
    # MediaPipe Face Mesh landmark indices
    NOSE_TIP = 4  # Nose tip (stable point)
    LEFT_EYE_INNER = 133  # Left eye inner corner
    LEFT_EYE_OUTER = 33  # Left eye outer corner
    RIGHT_EYE_INNER = 362  # Right eye inner corner
    RIGHT_EYE_OUTER = 263  # Right eye outer corner
    
    def __init__(self, grid_width: int = 8, grid_height: int = 10, grid_depth: int = 2):
        """
        Initialize the face grid tracker.
        
        Args:
            grid_width: Number of grid points horizontally (x-axis, default: 8)
            grid_height: Number of grid points vertically (y-axis, default: 10)
            grid_depth: Number of grid points in depth (z-axis, default: 2)
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_depth = grid_depth
        self.num_grid_points = grid_width * grid_height * grid_depth
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # Only track one face
            refine_landmarks=True,  # Use refined landmarks for better accuracy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Grid state
        self.grid_points_3d = None  # List of (x, y, z) tuples in normalized coordinates
        self.grid_points_2d = None  # List of (x, y) tuples in pixel coordinates (for visualization)
        self.hit_grid = None  # 3D boolean array tracking which points were hit [depth, height, width]
        self.frame_width = None
        self.frame_height = None
        
    def reset_hit_grid(self):
        """Reset the hit grid tracking (call at start of new capture)."""
        self.hit_grid = np.zeros((self.grid_depth, self.grid_height, self.grid_width), dtype=bool)
    
    def get_hit_grid_vector(self) -> List[int]:
        """
        Get the hit grid as a flattened 1D vector of 0s and 1s.
        
        Returns:
            List of integers (0 or 1) of length grid_width * grid_height * grid_depth
            Order: depth layers, then row by row (left to right, top to bottom)
        """
        if self.hit_grid is None:
            return [0] * self.num_grid_points
        
        # Flatten: for each depth layer, then row by row (left to right, top to bottom)
        return [1 if self.hit_grid[gz, gy, gx] else 0 
                for gz in range(self.grid_depth)
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
        for gz in range(self.grid_depth):
            for gy in range(self.grid_height):
                for gx in range(self.grid_width):
                    if self.hit_grid[gz, gy, gx]:
                        # Get the 3D coordinates of this grid point
                        idx = gz * (self.grid_width * self.grid_height) + gy * self.grid_width + gx
                        hit_points.append(self.grid_points_3d[idx])
        
        return hit_points
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Process a frame to detect face and build/update the grid.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            True if face detected and grid built, False otherwise
        """
        if self.face_mesh is None:
            return False
        
        # Store frame dimensions for coordinate conversion
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
        
        # Calculate eye centers (average of inner and outer corners)
        left_eye_center = (
            (left_eye_inner.x + left_eye_outer.x) / 2,
            (left_eye_inner.y + left_eye_outer.y) / 2,
            (left_eye_inner.z + left_eye_outer.z) / 2
        )
        right_eye_center = (
            (right_eye_inner.x + right_eye_outer.x) / 2,
            (right_eye_inner.y + right_eye_outer.y) / 2,
            (right_eye_inner.z + right_eye_outer.z) / 2
        )
        
        # Nose center (using nose tip)
        nose_center = (nose_tip.x, nose_tip.y, nose_tip.z)
        
        # Calculate horizontal spacing unit (distance from nose to left eye)
        dx = np.sqrt(
            (nose_center[0] - left_eye_center[0]) ** 2 +
            (nose_center[1] - left_eye_center[1]) ** 2 +
            (nose_center[2] - left_eye_center[2]) ** 2
        )
        
        # If distance is too small, use a default
        if dx < 1e-6:
            dx = 0.05  # Default spacing
        
        # Calculate vertical spacing (proportional to horizontal)
        # Use distance between eyes as reference for vertical scale
        eye_distance = np.sqrt(
            (left_eye_center[0] - right_eye_center[0]) ** 2 +
            (left_eye_center[1] - right_eye_center[1]) ** 2 +
            (left_eye_center[2] - right_eye_center[2]) ** 2
        )
        dy = eye_distance * 1.2  # Slightly larger for vertical spacing
        
        # Calculate depth spacing (z-axis)
        # Use a fraction of horizontal spacing for depth
        dz = dx * 0.8  # Depth spacing slightly smaller than width
        
        # Build the 3D grid
        # Grid extends ~4 points outside the head region on each side
        # Center the grid at nose center
        grid_points_3d = []
        grid_points_2d = []
        
        # Calculate grid bounds
        # Horizontal: center at nose, extend left and right
        # Vertical: center at nose, extend up and down
        # Depth: center at nose, extend forward and backward
        half_width = (self.grid_width - 1) / 2
        half_height = (self.grid_height - 1) / 2
        half_depth = (self.grid_depth - 1) / 2
        
        for gz in range(self.grid_depth):
            for gy in range(self.grid_height):
                for gx in range(self.grid_width):
                    # Calculate offset from center in 3D
                    offset_x = (gx - half_width) * dx
                    offset_y = (gy - half_height) * dy
                    offset_z = (gz - half_depth) * dz
                    
                    # Create grid point in 3D space
                    grid_x = nose_center[0] + offset_x
                    grid_y = nose_center[1] + offset_y
                    grid_z = nose_center[2] + offset_z  # Vary z for depth
                    
                    grid_points_3d.append((grid_x, grid_y, grid_z))
                    
                    # Convert to pixel coordinates for drawing (project 3D to 2D)
                    # Simple orthographic projection (ignoring z for 2D display)
                    pixel_x = int(grid_x * self.frame_width)
                    pixel_y = int(grid_y * self.frame_height)
                    grid_points_2d.append((pixel_x, pixel_y))
        
        self.grid_points_3d = grid_points_3d
        self.grid_points_2d = grid_points_2d
        
        return True
    
    def update_hit_grid(self, hand_landmarks_list: List[dict], hit_radius: float = 0.02):
        """
        Update the hit grid based on hand landmarks.
        
        For each hand landmark, find the closest grid point in 3D space and mark it as hit
        if within the hit radius.
        
        Args:
            hand_landmarks_list: List of dicts with 'landmarks' key containing
                                list of (x, y, z) tuples
            hit_radius: Distance threshold in normalized coordinates (default: 0.02)
        """
        if self.hit_grid is None:
            self.reset_hit_grid()
        
        if self.grid_points_3d is None:
            return
        
        # Process all hand landmarks
        for hand_data in hand_landmarks_list:
            landmarks = hand_data.get('landmarks', [])
            
            for landmark in landmarks:
                lx, ly, lz = landmark
                
                # Find closest grid point in 3D space
                min_dist = float('inf')
                closest_gx = 0
                closest_gy = 0
                closest_gz = 0
                
                for idx, (gx, gy, gz) in enumerate(self.grid_points_3d):
                    # Calculate 3D Euclidean distance
                    dist = np.sqrt(
                        (lx - gx) ** 2 +
                        (ly - gy) ** 2 +
                        (lz - gz) ** 2
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        # Convert linear index to 3D indices
                        closest_gz = idx // (self.grid_width * self.grid_height)
                        remainder = idx % (self.grid_width * self.grid_height)
                        closest_gy = remainder // self.grid_width
                        closest_gx = remainder % self.grid_width
                
                # Mark as hit if within radius
                if min_dist <= hit_radius:
                    self.hit_grid[closest_gz, closest_gy, closest_gx] = True
    
    def draw_grid(self, frame: np.ndarray, show_hits: bool = True) -> np.ndarray:
        """
        Draw the 3D grid on the frame (projected to 2D for visualization).
        
        Args:
            frame: BGR frame to draw on
            show_hits: If True, color hit points differently (green). 
                      If False, all points are gray regardless of hit status.
            
        Returns:
            Frame with grid drawn
        """
        if self.grid_points_2d is None:
            return frame
        
        annotated_frame = frame.copy()
        
        # Draw grid points (all depth layers, with different opacity/size for depth)
        for idx, (px, py) in enumerate(self.grid_points_2d):
            # Check if point is within frame bounds
            if 0 <= px < self.frame_width and 0 <= py < self.frame_height:
                # Convert linear index to 3D indices
                gz = idx // (self.grid_width * self.grid_height)
                remainder = idx % (self.grid_width * self.grid_height)
                gy = remainder // self.grid_width
                gx = remainder % self.grid_width
                
                # Determine color based on hit status and depth
                # IMPORTANT: Only show green highlights if show_hits is True AND the point was hit
                if show_hits and self.hit_grid is not None and self.hit_grid[gz, gy, gx]:
                    # Green for hit points, brighter for closer layers
                    depth_factor = 1.0 - (gz / max(self.grid_depth - 1, 1)) * 0.3
                    color = (0, int(255 * depth_factor), 0)
                    radius = 4  # Slightly larger for hit points
                else:
                    # Gray for all other points (unhit or show_hits=False)
                    # Vary by depth for visual distinction
                    depth_factor = 0.5 + (gz / max(self.grid_depth - 1, 1)) * 0.3
                    gray_val = int(88 * depth_factor)
                    color = (gray_val, gray_val, gray_val)
                    radius = 2  # Smaller for unhit points
                
                # Draw circle for grid point
                cv2.circle(annotated_frame, (px, py), radius, color, -1)
        
        # Draw grid lines connecting points in the same depth layer
        # Draw horizontal lines (within each depth layer)
        for gz in range(self.grid_depth):
            for gy in range(self.grid_height):
                for gx in range(self.grid_width - 1):
                    idx1 = gz * (self.grid_width * self.grid_height) + gy * self.grid_width + gx
                    idx2 = gz * (self.grid_width * self.grid_height) + gy * self.grid_width + gx + 1
                    pt1 = self.grid_points_2d[idx1]
                    pt2 = self.grid_points_2d[idx2]
                    if (0 <= pt1[0] < self.frame_width and 0 <= pt1[1] < self.frame_height and
                        0 <= pt2[0] < self.frame_width and 0 <= pt2[1] < self.frame_height):
                        # Lighter lines for front layers, darker for back
                        line_intensity = int(100 * (1.0 - gz / max(self.grid_depth - 1, 1) * 0.5))
                        cv2.line(annotated_frame, pt1, pt2, (line_intensity, line_intensity, line_intensity), 1)
        
        # Draw vertical lines (within each depth layer)
        for gz in range(self.grid_depth):
            for gx in range(self.grid_width):
                for gy in range(self.grid_height - 1):
                    idx1 = gz * (self.grid_width * self.grid_height) + gy * self.grid_width + gx
                    idx2 = gz * (self.grid_width * self.grid_height) + (gy + 1) * self.grid_width + gx
                    pt1 = self.grid_points_2d[idx1]
                    pt2 = self.grid_points_2d[idx2]
                    if (0 <= pt1[0] < self.frame_width and 0 <= pt1[1] < self.frame_height and
                        0 <= pt2[0] < self.frame_width and 0 <= pt2[1] < self.frame_height):
                        # Lighter lines for front layers, darker for back
                        line_intensity = int(100 * (1.0 - gz / max(self.grid_depth - 1, 1) * 0.5))
                        cv2.line(annotated_frame, pt1, pt2, (line_intensity, line_intensity, line_intensity), 1)
        
        return annotated_frame
    
    def release(self):
        """Release MediaPipe Face Mesh resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()


class HandCapture:
    """Manages webcam capture and MediaPipe hand detection."""
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize the hand capture system.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Grid / voxel tracking
        self.selected_landmark = 8  # Default: index_tip for path visualization
        
        # Use FaceGrid3D for 3D voxel grid (MediaPipe z-depth only)
        self.face_grid_tracker = FaceGrid3D(
            breadth=8,
            length=10,
            depth_layers=3,  # 3 layers: near, middle, far
            track_landmark_paths=True,
            tracked_landmarks=[8, 4]  # index_tip, thumb_tip
        )
        
    def initialize(self) -> bool:
        """
        Initialize the camera and MediaPipe Hands.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Initialize MediaPipe Hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,  # Detect up to 2 hands
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
            
            return True
        except Exception as e:
            print(f"Error initializing capture: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the webcam.
        
        Returns:
            BGR frame as numpy array, or None if failed
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        track_grid_hits: bool = False,
        draw_grid: bool = True,
        show_hits: bool = True
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Process a frame to detect hand landmarks.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            Tuple of (annotated_frame, landmarks_list)
            landmarks_list contains dicts with keys:
                - 'hand': 'left' or 'right'
                - 'landmarks': list of (x, y, z) tuples (21 points per hand)
        """
        if self.hands is None:
            annotated_frame = frame.copy()
            # Still try to process face for grid
            if draw_grid:
                self.face_grid_tracker.process_frame(frame)
                annotated_frame = self.face_grid_tracker.draw_grid(
                    annotated_frame,
                    show_hits=show_hits,
                    selected_landmark=getattr(self, 'selected_landmark', 8),
                    show_indices=True
                )
            return annotated_frame, []
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for drawing
        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        landmarks_list = []
        
        # Extract landmarks for each detected hand
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Determine which hand (left/right from camera's perspective)
                hand_label = handedness.classification[0].label
                
                # Extract 3D coordinates
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    # MediaPipe provides normalized coordinates (0-1) and z depth
                    landmarks_3d.append((
                        landmark.x,  # Normalized x (0-1)
                        landmark.y,  # Normalized y (0-1)
                        landmark.z   # Depth (relative, can be negative)
                    ))
                
                landmarks_list.append({
                    'hand': hand_label.lower(),  # 'Left' or 'Right'
                    'landmarks': landmarks_3d
                })
                
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Process face for grid (must be called every frame to update grid position)
        if draw_grid:
            self.face_grid_tracker.process_frame(frame)
        
        # Update hit grid if tracking (always use 3D voxel tracking)
        if track_grid_hits and landmarks_list:
            self.face_grid_tracker.update_hit_tracking(landmarks_list)
        
        # Draw grid on top (always 3D voxel grid)
        if draw_grid:
            # Get selected landmark for path visualization (default: index_tip = 8)
            selected_landmark = getattr(self, 'selected_landmark', 8)
            annotated_frame = self.face_grid_tracker.draw_grid(
                annotated_frame,
                show_hits=show_hits,
                selected_landmark=selected_landmark,
                show_indices=False
            )
        
        return annotated_frame, landmarks_list
    
    def get_landmarks(self, frame: np.ndarray) -> List[dict]:
        """
        Extract hand landmarks from a frame without drawing.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            List of dicts with 'hand' and 'landmarks' keys
        """
        if self.hands is None:
            return []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process
        results = self.hands.process(rgb_frame)
        
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = handedness.classification[0].label
                
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:
                    landmarks_3d.append((landmark.x, landmark.y, landmark.z))
                
                landmarks_list.append({
                    'hand': hand_label.lower(),
                    'landmarks': landmarks_3d
                })
        
        return landmarks_list
    
    def start_grid_tracking(self):
        """Start a new grid tracking session (call when Start Capture is pressed)."""
        self.face_grid_tracker.reset_hit_tracking()
    
    def stop_grid_tracking(self):
        """Stop grid tracking and reset hits (call when Stop Capture is pressed)."""
        # Reset hit grid after capture stops
        self.face_grid_tracker.reset_hit_tracking()
    
    def get_hit_grid_vector(self) -> List[int]:
        """
        Get the current hit grid as a flattened vector.
        
        Returns:
            List of 0s and 1s representing which grid points were hit
        """
        return self.face_grid_tracker.get_hit_grid_3d()
    
    def get_hit_points_coordinates(self) -> List[Tuple[float, float, float]]:
        """
        Get the 3D coordinates of all hit grid points.
        
        Returns:
            List of (x, y, z) tuples for each grid point that was hit
        """
        return self.face_grid_tracker.get_voxel_hit_centers()
    
    def get_voxel_paths(self) -> dict:
        """
        Get ordered voxel paths per landmark.
        
        Returns:
            Dictionary mapping landmark names to ordered lists of voxel indices
        """
        if hasattr(self.face_grid_tracker, 'get_voxel_paths'):
            return self.face_grid_tracker.get_voxel_paths()
        return {}
    
    def get_hit_order(self) -> List[int]:
        """
        Get the ordered voxel indices representing first-hit sequence.
        """
        if hasattr(self.face_grid_tracker, 'get_hit_order'):
            return self.face_grid_tracker.get_hit_order()
        return []
    
    def get_num_grid_points(self) -> int:
        """Get the total number of grid points."""
        return self.face_grid_tracker.num_voxels
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
        if self.hands is not None:
            self.hands.close()

