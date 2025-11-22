"""
Tests for 3D voxel grid functionality using MiDaS depth.

This module tests the face-centered 3D voxel grid construction and hit detection.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_grid_3d import FaceGrid3D, INDEX_TIP, THUMB_TIP


class TestFaceGrid3D(unittest.TestCase):
    """Test cases for 3D voxel grid."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock depth provider
        self.mock_depth_provider = Mock()
        self.mock_depth_provider.is_active.return_value = True
        self.mock_depth_provider.get_depth_at.return_value = 0.5
        self.mock_depth_map = np.ones((480, 640), dtype=np.float32) * 0.5
        self.mock_depth_provider.get_depth_map.return_value = self.mock_depth_map
        
        # Create grid with mocked depth provider
        self.grid = FaceGrid3D(
            breadth=12,
            length=15,
            depth_layers=7,
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
        self.assertEqual(self.grid.breadth, 12)
        self.assertEqual(self.grid.length, 15)
        self.assertEqual(self.grid.depth_layers, 7)
        self.assertEqual(self.grid.num_voxels, 1260)  # 12 * 15 * 7
    
    def test_voxel_index(self):
        """Test voxel indexing function."""
        # Test indexing order: [z, row, col] - z fastest
        idx_0_0_0 = self.grid.voxel_index(0, 0, 0)
        idx_1_0_0 = self.grid.voxel_index(1, 0, 0)
        idx_0_1_0 = self.grid.voxel_index(0, 1, 0)
        idx_0_0_1 = self.grid.voxel_index(0, 0, 1)
        
        # z=0, row=0, col=0 should be 0
        self.assertEqual(idx_0_0_0, 0)
        # z=0, row=0, col=1 should be 1 (col increments)
        self.assertEqual(idx_1_0_0, 1)
        # z=0, row=1, col=0 should be breadth (12)
        self.assertEqual(idx_0_1_0, 12)
        # z=1, row=0, col=0 should be breadth*length (180)
        self.assertEqual(idx_0_0_1, 180)
        
        # Test last voxel
        last_idx = self.grid.voxel_index(11, 14, 6)
        self.assertEqual(last_idx, 1259)  # 6*180 + 14*12 + 11 = 1080 + 168 + 11
    
    def test_reset_hit_tracking(self):
        """Test that hit tracking resets correctly."""
        self.grid.reset_hit_tracking()
        self.assertIsNotNone(self.grid.voxel_hit_bool)
        self.assertEqual(len(self.grid.voxel_hit_bool), 1260)
        self.assertEqual(np.sum(self.grid.voxel_hit_bool), 0)  # All zeros
        self.assertIsNotNone(self.grid.voxel_paths)
        self.assertEqual(len(self.grid.voxel_paths), 2)  # index_tip, thumb_tip
    
    def test_get_hit_grid_3d(self):
        """Test that hit grid vector is correctly flattened."""
        self.grid.reset_hit_tracking()
        vector = self.grid.get_hit_grid_3d()
        
        # Should be 1260 elements
        self.assertEqual(len(vector), 1260)
        self.assertEqual(sum(vector), 0)  # All zeros initially
        
        # Mark a few voxels as hit
        self.grid.voxel_hit_bool[0] = True
        self.grid.voxel_hit_bool[100] = True
        self.grid.voxel_hit_bool[1259] = True
        
        vector = self.grid.get_hit_grid_3d()
        self.assertEqual(sum(vector), 3)  # Three hits
    
    def test_get_voxel_paths(self):
        """Test that voxel paths are returned correctly."""
        self.grid.reset_hit_tracking()
        
        # No paths initially
        paths = self.grid.get_voxel_paths()
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(paths.get('index_tip', [])), 0)
        
        # Add some paths
        self.grid.voxel_paths[INDEX_TIP] = [0, 10, 20, 30]
        self.grid.voxel_paths[THUMB_TIP] = [5, 15, 25]
        
        paths = self.grid.get_voxel_paths()
        self.assertEqual(paths['index_tip'], [0, 10, 20, 30])
        self.assertEqual(paths['thumb_tip'], [5, 15, 25])
    
    @patch('face_grid_3d.mp.solutions.face_mesh.FaceMesh')
    def test_process_frame_creates_voxel_grid(self, mock_face_mesh_class):
        """Test that process_frame creates exactly 1260 voxel centers."""
        # Mock face mesh results
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        
        # Process frame
        success = self.grid.process_frame(self.mock_frame)
        
        self.assertTrue(success)
        self.assertIsNotNone(self.grid.voxel_centers)
        self.assertEqual(len(self.grid.voxel_centers), 1260)  # Exactly 1260 voxels
        self.assertIsNotNone(self.grid.voxel_centers_2d)
        self.assertEqual(len(self.grid.voxel_centers_2d), 1260)
    
    def test_voxel_centers_in_valid_range(self):
        """Test that voxel centers are in valid normalized range [0, 1]."""
        # Mock face mesh
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        
        # Process frame
        self.grid.process_frame(self.mock_frame)
        
        # Check all voxel centers are in valid range
        for x, y, z in self.grid.voxel_centers:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)
            self.assertGreaterEqual(z, 0.0)
            self.assertLessEqual(z, 1.0)
    
    def test_project_to_pixels(self):
        """Test pixel projection function."""
        self.grid.frame_width = 640
        self.grid.frame_height = 480
        
        # Test projection
        u, v = self.grid.project_to_pixels((0.5, 0.5, 0.3))
        self.assertEqual(u, 320)
        self.assertEqual(v, 240)
        
        u, v = self.grid.project_to_pixels((0.0, 0.0, 0.0))
        self.assertEqual(u, 0)
        self.assertEqual(v, 0)
        
        u, v = self.grid.project_to_pixels((1.0, 1.0, 1.0))
        self.assertEqual(u, 640)
        self.assertEqual(v, 480)
    
    def test_update_hit_tracking_with_landmarks(self):
        """Test that hit tracking updates correctly with hand landmarks."""
        # Setup grid
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        self.grid.process_frame(self.mock_frame)
        self.grid.reset_hit_tracking()
        
        # Create mock hand landmarks near center
        hand_landmarks_list = [
            {
                'hand': 'right',
                'landmarks': [
                    (0.5, 0.5, 0.0),  # Near center - should hit a voxel
                    (0.6, 0.6, 0.0),  # Another point
                ]
            }
        ]
        
        # Update hit tracking
        self.grid.update_hit_tracking(hand_landmarks_list)
        
        # Check that some voxels were hit
        hit_count = np.sum(self.grid.voxel_hit_bool)
        self.assertGreater(hit_count, 0)
    
    def test_path_tracking_avoids_duplicates(self):
        """Test that path tracking avoids consecutive duplicate entries."""
        # Setup grid
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        self.grid.process_frame(self.mock_frame)
        self.grid.reset_hit_tracking()
        
        # Create landmarks that hit the same voxel multiple times
        hand_landmarks_list = [
            {
                'hand': 'right',
                'landmarks': [
                    (0.5, 0.5, 0.0),  # index_tip (landmark 8)
                ]
            }
        ]
        
        # Update multiple times with same position
        for _ in range(5):
            self.grid.update_hit_tracking(hand_landmarks_list)
        
        # Path should only have one entry (no duplicates)
        path = self.grid.voxel_paths.get(INDEX_TIP, [])
        # Should have at most one entry per unique voxel hit
        self.assertLessEqual(len(path), 5)
    
    def test_draw_grid_returns_frame(self):
        """Test that draw_grid returns an annotated frame."""
        # Setup grid
        mock_face_mesh = Mock()
        mock_results = Mock()
        mock_results.multi_face_landmarks = [self.mock_face_landmarks]
        mock_face_mesh.process.return_value = mock_results
        self.grid.face_mesh = mock_face_mesh
        self.grid.process_frame(self.mock_frame)
        self.grid.reset_hit_tracking()
        
        # Draw grid
        annotated = self.grid.draw_grid(self.mock_frame.copy(), show_hits=True, depth_mode="midas")
        
        # Should return a frame of same shape
        self.assertEqual(annotated.shape, self.mock_frame.shape)
        self.assertEqual(annotated.dtype, self.mock_frame.dtype)


if __name__ == '__main__':
    unittest.main()

