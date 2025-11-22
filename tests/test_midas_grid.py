"""
Tests for MiDaS-based face grid functionality.

This module tests the face-centered 3D grid construction using MiDaS depth.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_grid_midas import FaceGridMiDaS


class TestMiDaSGrid(unittest.TestCase):
    """Test cases for MiDaS-based face grid."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock depth provider
        self.mock_depth_provider = Mock()
        self.mock_depth_provider.is_active.return_value = True
        self.mock_depth_provider.get_depth_at.return_value = 0.5
        self.mock_depth_map = np.ones((480, 640), dtype=np.float32) * 0.5
        self.mock_depth_provider.get_depth_map.return_value = self.mock_depth_map
        
        # Create grid with mocked depth provider
        self.grid = FaceGridMiDaS(
            grid_width=12,
            grid_height=15,
            depth_provider=self.mock_depth_provider
        )
        
        # Mock frame
        self.mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock MediaPipe face landmarks
        self.mock_face_landmarks = Mock()
        self.mock_face_landmarks.landmark = []
        
        # Create mock landmarks for nose and eyes
        for i in range(500):  # MediaPipe has 468 landmarks
            landmark = Mock()
            if i == 4:  # NOSE_TIP
                landmark.x = 0.5
                landmark.y = 0.5
                landmark.z = 0.0
            elif i == 133:  # LEFT_EYE_INNER
                landmark.x = 0.45
                landmark.y = 0.48
                landmark.z = 0.0
            elif i == 33:  # LEFT_EYE_OUTER
                landmark.x = 0.42
                landmark.y = 0.48
                landmark.z = 0.0
            elif i == 362:  # RIGHT_EYE_INNER
                landmark.x = 0.55
                landmark.y = 0.48
                landmark.z = 0.0
            elif i == 263:  # RIGHT_EYE_OUTER
                landmark.x = 0.58
                landmark.y = 0.48
                landmark.z = 0.0
            else:
                landmark.x = 0.0
                landmark.y = 0.0
                landmark.z = 0.0
            self.mock_face_landmarks.landmark.append(landmark)
    
    def test_grid_initialization(self):
        """Test that grid initializes with correct dimensions."""
        self.assertEqual(self.grid.grid_width, 12)
        self.assertEqual(self.grid.grid_height, 15)
        self.assertEqual(self.grid.num_grid_points, 180)  # 12 * 15
    
    def test_reset_hit_grid(self):
        """Test that hit grid resets correctly."""
        self.grid.reset_hit_grid()
        self.assertIsNotNone(self.grid.hit_grid)
        self.assertEqual(self.grid.hit_grid.shape, (15, 12))
        self.assertEqual(np.sum(self.grid.hit_grid), 0)  # All zeros
    
    def test_get_hit_grid_vector(self):
        """Test that hit grid vector is correctly flattened."""
        self.grid.reset_hit_grid()
        vector = self.grid.get_hit_grid_vector()
        
        # Should be 180 elements (12 * 15)
        self.assertEqual(len(vector), 180)
        self.assertEqual(sum(vector), 0)  # All zeros initially
        
        # Mark a few points as hit
        self.grid.hit_grid[0, 0] = True
        self.grid.hit_grid[5, 3] = True
        self.grid.hit_grid[14, 11] = True
        
        vector = self.grid.get_hit_grid_vector()
        self.assertEqual(sum(vector), 3)  # Three hits
    
    def test_get_hit_points_coordinates(self):
        """Test that hit points coordinates are returned correctly."""
        self.grid.reset_hit_grid()
        self.grid.grid_points_3d = [(0.1, 0.2, 0.3)] * 180
        
        # No hits initially
        hit_points = self.grid.get_hit_points_coordinates()
        self.assertEqual(len(hit_points), 0)
        
        # Mark some points as hit
        self.grid.hit_grid[0, 0] = True
        self.grid.hit_grid[1, 1] = True
        
        hit_points = self.grid.get_hit_points_coordinates()
        self.assertEqual(len(hit_points), 2)
    
    @patch('face_grid_midas.mp.solutions.face_mesh.FaceMesh')
    def test_process_frame_creates_grid(self, mock_face_mesh_class):
        """Test that process_frame creates exactly 180 grid points."""
        # Mock face mesh results
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        
        # Process frame
        success = self.grid.process_frame(self.mock_frame)
        
        self.assertTrue(success)
        self.assertIsNotNone(self.grid.grid_points_3d)
        self.assertEqual(len(self.grid.grid_points_3d), 180)  # Exactly 180 points
        self.assertIsNotNone(self.grid.grid_points_2d)
        self.assertEqual(len(self.grid.grid_points_2d), 180)  # Exactly 180 pixel points
    
    def test_grid_points_in_valid_range(self):
        """Test that grid points are in valid normalized range [0, 1]."""
        # Mock face mesh
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        
        # Process frame
        self.grid.process_frame(self.mock_frame)
        
        # Check all points are in valid range
        for x, y, z in self.grid.grid_points_3d:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)
            self.assertGreaterEqual(z, 0.0)
            self.assertLessEqual(z, 1.0)
    
    def test_pixel_projection_in_frame_bounds(self):
        """Test that pixel projections are within frame bounds."""
        # Mock face mesh
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        
        # Process frame
        self.grid.process_frame(self.mock_frame)
        
        # Check pixel coordinates are reasonable (may be slightly outside due to rounding)
        for px, py in self.grid.grid_points_2d:
            # Allow some margin for rounding
            self.assertGreater(px, -10)
            self.assertLess(px, 650)
            self.assertGreater(py, -10)
            self.assertLess(py, 490)
    
    def test_update_hit_grid_with_landmarks(self):
        """Test that hit grid updates correctly with hand landmarks."""
        # Setup grid
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        self.grid.process_frame(self.mock_frame)
        self.grid.reset_hit_grid()
        
        # Create mock hand landmarks
        hand_landmarks_list = [
            {
                'hand': 'right',
                'landmarks': [
                    (0.5, 0.5, 0.0),  # Near center
                    (0.6, 0.6, 0.0),  # Another point
                ]
            }
        ]
        
        # Update hit grid
        self.grid.update_hit_grid(hand_landmarks_list, hit_radius=0.1)
        
        # Check that some points were hit
        hit_count = np.sum(self.grid.hit_grid)
        self.assertGreater(hit_count, 0)
    
    def test_draw_grid_returns_frame(self):
        """Test that draw_grid returns an annotated frame."""
        # Setup grid
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        self.grid.process_frame(self.mock_frame)
        self.grid.reset_hit_grid()
        
        # Draw grid
        annotated = self.grid.draw_grid(self.mock_frame.copy(), show_hits=True, depth_mode="midas")
        
        # Should return a frame of same shape
        self.assertEqual(annotated.shape, self.mock_frame.shape)
        self.assertEqual(annotated.dtype, self.mock_frame.dtype)


if __name__ == '__main__':
    unittest.main()

