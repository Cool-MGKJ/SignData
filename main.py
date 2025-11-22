"""
Main entry point for ASL Dataset Collection Tool.

This application allows users to collect labelled ASL sign samples
using MediaPipe hand tracking and a webcam.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
from pathlib import Path

from capture import HandCapture
from normalization import normalize_multiple_hands, get_hand_info
from dataset_io import DatasetManager
from ui import ASLDataCollectionUI


class ASLDataCollectionApp:
    """Main application class that coordinates all components."""
    
    def __init__(self):
        """Initialize the application."""
        self.root = tk.Tk()
        self.ui = ASLDataCollectionUI(self.root)
        
        # Initialize components
        self.capture = HandCapture(camera_index=0)
        self.dataset = DatasetManager()
        
        # Configuration
        self.num_points_per_hand = 21  # MediaPipe provides 21 landmarks per hand
        self.spacing_ratio = 1.0  # Normalization scaling factor
        self.depth_mode = "off"  # Current depth mode
        
        # State
        self.is_capturing = False
        self.captured_frame = None
        self.captured_landmarks = None
        self.captured_hit_grid = None
        self.captured_hit_points = None  # List of (x, y, z) tuples for hit points
        self.captured_hit_grid_midas = None  # MiDaS hit grid vector
        self.captured_hit_points_midas = None  # MiDaS hit points coordinates
        
        # Setup callbacks
        self.ui.set_callbacks(
            on_start_capture=self.start_capture,
            on_stop_capture=self.stop_capture,
            on_save_sample=self.save_sample,
            on_export_dataset=self.export_dataset,
            on_clear_session=self.clear_session,
            on_depth_mode_change=self.handle_depth_mode_change
        )
        
        # Initialize depth mode from UI
        self.depth_mode = self.ui.get_depth_mode()
        self.capture.set_depth_mode(self.depth_mode)
        self.ui.update_midas_status(self.capture.is_midas_active())
        
        # Initialize camera
        if not self.capture.initialize():
            messagebox.showerror(
                "Error",
                "Failed to initialize camera. Please check if a webcam is connected."
            )
            self.root.after(100, self.root.destroy)
            return
        
        # Start video loop
        self.update_video()
    
    def start_capture(self):
        """Start capture mode."""
        self.is_capturing = True
        # Reset grid tracking for new capture session
        self.capture.start_grid_tracking()
        self.captured_hit_grid = None
        self.captured_hit_points = None
        self.captured_hit_grid_midas = None
        self.captured_hit_points_midas = None
    
    def stop_capture(self):
        """Stop capture and process the current frame."""
        # IMPORTANT: Set is_capturing to False FIRST, before getting data
        # This ensures the grid visualization stops showing hits immediately
        self.is_capturing = False
        
        if self.captured_frame is None:
            messagebox.showwarning("Warning", "No frame captured. Please try again.")
            # Reset grid even if no frame
            self.capture.stop_grid_tracking()
            return
        
        # Extract landmarks from the captured frame (for the final hand shape)
        landmarks_list = self.capture.get_landmarks(self.captured_frame)
        
        if not landmarks_list:
            messagebox.showwarning(
                "Warning",
                "No hands detected in the captured frame. Please ensure your hands are visible."
            )
            # Reset grid even if no hands detected
            self.capture.stop_grid_tracking()
            return
        
        # Store for saving
        self.captured_landmarks = landmarks_list
        
        # Get the hit points with their 3D coordinates (accumulated during the entire capture session)
        # Do this BEFORE resetting the grid
        self.captured_hit_points = self.capture.get_hit_points_coordinates()
        
        # Also get the binary vector for compatibility
        self.captured_hit_grid = self.capture.get_hit_grid_vector()
        
        # Get 3D voxel grid data (always available now, regardless of depth mode)
        self.captured_hit_grid_midas = self.capture.get_hit_grid_vector()  # 3D voxel grid (1260)
        self.captured_hit_points_midas = self.capture.get_hit_points_coordinates()
        
        # Reset grid tracking (so hits don't show after stop)
        # This clears the hit_grid so visualization shows no green points
        self.capture.stop_grid_tracking()
        
        # Normalize landmarks (for conventional hand-shape representation)
        hand_info = get_hand_info(landmarks_list)
        normalized_points = normalize_multiple_hands(
            landmarks_list,
            self.num_points_per_hand,
            self.spacing_ratio
        )
        
        if normalized_points is None:
            messagebox.showerror("Error", "Failed to normalize landmarks.")
            return
        
        # Display in UI (with hit grid info)
        num_hit_points = len(self.captured_hit_points) if self.captured_hit_points else 0
        total_grid_points = self.capture.get_num_grid_points()
        
        # Update hit count display (always 3D voxel grid: 1260 voxels)
        num_hit_midas = sum(self.captured_hit_grid_midas) if self.captured_hit_grid_midas else 0
        self.ui.update_hit_count(num_hit_midas, 1260)
        
        self.ui.display_recent_sample(
            normalized_points, 
            hand_info, 
            hit_grid_info=(num_hit_points, total_grid_points)
        )
    
    def save_sample(self, label: str) -> bool:
        """
        Save the current sample to the dataset.
        
        Args:
            label: The ASL sign label
        
        Returns:
            True if successful, False otherwise
        """
        if self.captured_landmarks is None:
            return False
        
        # Normalize landmarks (conventional hand-shape representation)
        hand_info = get_hand_info(self.captured_landmarks)
        normalized_points = normalize_multiple_hands(
            self.captured_landmarks,
            self.num_points_per_hand,
            self.spacing_ratio
        )
        
        if normalized_points is None:
            return False
        
        # Get hit points with coordinates (which points were hit during capture)
        hit_points = self.captured_hit_points if self.captured_hit_points is not None else []
        
        # Also keep binary vector for backward compatibility
        hit_grid_vector = self.captured_hit_grid if self.captured_hit_grid is not None else []
        
        # Get MiDaS-specific data (for backward compatibility)
        hit_grid_midas = self.captured_hit_grid_midas if self.captured_hit_grid_midas is not None else []
        hit_points_midas = self.captured_hit_points_midas if self.captured_hit_points_midas is not None else []
        
        # Get 3D voxel grid data (always available now, regardless of depth mode)
        grid_center_z = self.capture.get_grid_center_z()
        grid_spacing_norm = self.capture.get_grid_spacing_norm()
        hit_grid_3d = self.captured_hit_grid_midas if self.captured_hit_grid_midas is not None else []
        voxel_paths = self.capture.get_voxel_paths()
        
        # Get grid dimensions from tracker
        grid_dims = [12, 15, 7]  # Default
        if hasattr(self.capture.face_grid_tracker, 'breadth'):
            grid_dims = [
                self.capture.face_grid_tracker.breadth,
                self.capture.face_grid_tracker.length,
                self.capture.face_grid_tracker.depth_layers
            ]
        
        # Add to dataset (with both hand-shape and hit-grid data)
        # IMPORTANT: The 'points' field remains unchanged - MediaPipe normalized hand landmarks
        sample_id = self.dataset.add_sample(
            label=label,
            normalized_points=normalized_points,  # MediaPipe points - UNCHANGED
            hand=hand_info,
            hit_grid_vector=hit_grid_vector,
            hit_points_coordinates=hit_points,
            hit_grid_midas=hit_grid_midas,
            depth_mode=self.depth_mode,
            grid_center_z=grid_center_z,
            grid_spacing_norm=grid_spacing_norm,
            hit_points_midas=hit_points_midas,
            hit_grid_3d=hit_grid_3d,
            voxel_paths=voxel_paths,
            grid_dims=grid_dims
        )
        
        # Update UI
        self.ui.add_sample_to_table(
            sample_id,
            label,
            hand_info,
            len(normalized_points)
        )
        self.ui.update_sample_count(self.dataset.get_sample_count())
        
        # Save to file automatically
        self._auto_save_sample()
        
        # Reset captured data
        self.captured_landmarks = None
        self.captured_frame = None
        self.captured_hit_grid = None
        self.captured_hit_points = None
        self.captured_hit_grid_midas = None
        self.captured_hit_points_midas = None
        
        return True
    
    def _auto_save_sample(self):
        """Automatically save the latest sample to disk."""
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save as JSON (append mode)
        json_path = data_dir / "asl_dataset.json"
        self.dataset.save_as_json(str(json_path), append=True)
    
    def export_dataset(self):
        """Export the full dataset to a file."""
        if self.dataset.get_sample_count() == 0:
            messagebox.showwarning("Warning", "No samples to export.")
            return
        
        # Ask user for file format and location
        file_path = filedialog.asksaveasfilename(
            title="Export Dataset",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Export based on file extension
        if file_path.endswith('.json'):
            self.dataset.save_as_json(file_path, append=False)
        elif file_path.endswith('.csv'):
            self.dataset.save_as_csv(file_path, append=False)
        else:
            # Default to JSON
            file_path += '.json'
            self.dataset.save_as_json(file_path, append=False)
        
        messagebox.showinfo("Success", f"Dataset exported to {file_path}")
    
    def clear_session(self):
        """Clear all samples from the current session."""
        self.dataset.clear()
        self.captured_landmarks = None
        self.captured_frame = None
        self.captured_hit_grid = None
        self.captured_hit_points = None
        self.captured_hit_grid_midas = None
        self.captured_hit_points_midas = None
        self.ui.update_sample_count(0)
    
    def handle_depth_mode_change(self, new_mode: str):
        """
        Handle depth mode change from UI.
        
        Args:
            new_mode: New depth mode ("off" or "midas")
        """
        self.depth_mode = new_mode
        self.capture.set_depth_mode(new_mode)
        self.ui.update_midas_status(self.capture.is_midas_active())
    
    def update_video(self):
        """Update the video feed (called repeatedly)."""
        frame = self.capture.read_frame()
        
        if frame is not None:
            # Process frame for display (with landmarks drawn and grid tracking)
            # Track grid hits and show hits only during capture, always draw grid
            annotated_frame, landmarks_list = self.capture.process_frame(
                frame,
                track_grid_hits=self.is_capturing,  # Track hits only during capture
                draw_grid=True,  # Always show grid
                show_hits=self.is_capturing,  # Show hit highlights only during capture
                depth_mode=self.depth_mode  # Pass depth mode for visualization
            )
            
            # Update MiDaS status and hit count during capture
            if self.is_capturing:
                self.ui.update_midas_status(self.capture.is_midas_active())
                hit_grid_midas = self.capture.get_hit_grid_vector()
                num_hit = sum(hit_grid_midas) if hit_grid_midas else 0
                self.ui.update_hit_count(num_hit, 1260)  # 3D voxel grid: 12*15*7 = 1260
            
            # If capturing, store the current frame
            if self.is_capturing:
                self.captured_frame = frame.copy()
            
            # Update UI
            self.ui.update_camera_frame(annotated_frame)
        
        # Schedule next update
        self.root.after(30, self.update_video)  # ~30 FPS
    
    def run(self):
        """Start the application main loop."""
        try:
            self.root.mainloop()
        finally:
            # Cleanup
            self.capture.release()
            cv2.destroyAllWindows()


def main():
    """Entry point for the application."""
    app = ASLDataCollectionApp()
    app.run()


if __name__ == "__main__":
    main()

