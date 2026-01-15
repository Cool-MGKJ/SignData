"""
Main entry point for ASL Dataset Collection Tool.

This application allows users to collect labelled ASL sign samples
using MediaPipe hand tracking and a webcam.

Developer Notes:
- Coordinates the UI (`ASLDataCollectionUI`), capture (`HandCapture`),
  normalization (`normalize_hand_data`), and dataset storage (`DatasetManager`).
- Live normalization is applied when saving samples; there is no offline
  normalization pass in this project.
- Key runtime flow: initialize capture -> start video loop -> start/stop capture
  -> save samples to in-memory dataset -> export via UI.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
from pathlib import Path

from capture import HandCapture
from normalization import normalize_multiple_hands, get_hand_info, standardize_features
from dataset_io import DatasetManager
from ui import ASLDataCollectionUI
import numpy as np


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
        
        # State
        self.is_capturing = False
        self.captured_frame = None
        self.captured_landmarks = None
        self.captured_hit_order = None
        self.captured_chain_code = None  # Chain code captured during gesture
        self.captured_palm_angles_left = None  # Palm angles for left hand
        self.captured_palm_angles_right = None  # Palm angles for right hand
        self.captured_trigger_distance_left = None  # Trigger distances for left hand
        self.captured_trigger_distance_right = None  # Trigger distances for right hand
        self.last_known_landmarks = None  # Track last known hand posture during capture
        
        # Setup callbacks
        self.ui.set_callbacks(
            on_start_capture=self.start_capture,
            on_stop_capture=self.stop_capture,
            on_save_sample=self.save_sample,
            on_export_dataset=self.export_dataset,
            on_clear_session=self.clear_session
        )
        
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
        self.captured_hit_order = None
        self.last_known_landmarks = None  # Reset last known landmarks at start of capture
    
    def stop_capture(self):
        """Stop capture and process the current frame."""
        if self.captured_frame is None:
            messagebox.showwarning("Warning", "No frame captured. Please try again.")
            return
        
        # Extract landmarks from the captured frame
        landmarks_list = self.capture.get_landmarks(self.captured_frame)
        
        # Use current frame landmarks if available, otherwise use last known landmarks
        # This preserves the hand posture even if hand moved out of frame
        if landmarks_list and len(landmarks_list) > 0:
            # Current frame has hands detected - use these
            self.captured_landmarks = landmarks_list
        elif self.last_known_landmarks and len(self.last_known_landmarks) > 0:
            # No hands in current frame, but we have last known posture - use that
            print("No hands detected in current frame, using last known hand posture")
            self.captured_landmarks = self.last_known_landmarks
        else:
            # No hands detected at all during capture session
            self.captured_landmarks = []
        
        # Get the hit order and chain code (accumulated during the entire capture session)
        # Do this BEFORE resetting the grid
        self.captured_hit_order = self.capture.get_hit_order()
        self.captured_chain_code = self.capture.get_chain_code()
        
        # Get palm angles and trigger distances for both hands
        self.captured_palm_angles_left = self.capture.get_palm_angles_left()
        self.captured_palm_angles_right = self.capture.get_palm_angles_right()
        self.captured_trigger_distance_left = self.capture.get_trigger_distance_left()
        self.captured_trigger_distance_right = self.capture.get_trigger_distance_right()
        
        # Reset grid tracking (so hits don't show after stop)
        # This clears the hit_grid so visualization shows no green points
        self.capture.stop_grid_tracking()
        
        # Apply "Standard Hand" normalization for display
        # The captured landmarks are in raw MediaPipe format (for grid operations)
        # Normalize them here for consistent visualization
        if self.captured_landmarks:
            hand_info = get_hand_info(self.captured_landmarks)
            # Normalize each hand to "Standard Hand" format for display
            normalized_points = []
            for hand_data in self.captured_landmarks:
                landmarks = hand_data.get('landmarks', [])
                if len(landmarks) == 21:
                    from normalization import normalize_hand_data
                    normalized = normalize_hand_data(landmarks, apply_rotation=True)
                    normalized_points.extend(normalized)
            if not normalized_points:
                normalized_points = []
                hand_info = "none"
        else:
            normalized_points = []
            hand_info = "none"
        
        # Display in UI
        total_grid_points = self.capture.get_num_grid_points()
        num_hit = len(self.captured_hit_order) if self.captured_hit_order else 0
        
        # Update hit count display using current grid dimensions
        self.ui.update_hit_count(num_hit, total_grid_points)
        
        # Debug: Print what we're passing to display
        print(f"Displaying sample: {len(normalized_points)} points, hand={hand_info}")
        if normalized_points and len(normalized_points) > 0:
            print(f"First point: {normalized_points[0]}")
        
        self.ui.display_recent_sample(
            normalized_points, 
            hand_info
        )
    
    def save_sample(self, label: str) -> bool:
        """
        Save the current sample to the dataset.
        
        Args:
            label: The ASL sign label
        
        Returns:
            True if successful, False otherwise
        """
        # Separate left and right hand points
        points_left = []
        points_right = []
        
        if self.captured_landmarks is None or len(self.captured_landmarks) == 0:
            # No landmarks captured
            points_left = []
            points_right = []
        else:
            # Apply "Standard Hand" normalization when saving to dataset
            # This ensures consistent representation for PCA/clustering
            for hand_data in self.captured_landmarks:
                landmarks = hand_data.get('landmarks', [])
                handedness = hand_data.get('handedness', 'Unknown')
                
                # Debug: Print what we're processing
                print(f"Processing hand: handedness={handedness}, num_landmarks={len(landmarks)}")
                
                if len(landmarks) == 21:
                    # Apply Standard Hand normalization
                    from normalization import normalize_hand_data
                    normalized = normalize_hand_data(landmarks, apply_rotation=True)
                    
                    # Debug: Print what we normalized
                    print(f"Normalized {handedness} hand: {len(normalized)} points")
                    
                    # Store in appropriate left/right array
                    if handedness == 'Left':
                        points_left = normalized
                    elif handedness == 'Right':
                        points_right = normalized
                    else:
                        # Unknown handedness - try to store it somewhere
                        print(f"WARNING: Unknown handedness '{handedness}', storing as right hand")
                        if not points_right:  # If right not set, use this
                            points_right = normalized
        
        hit_order = self.captured_hit_order if self.captured_hit_order is not None else []
        chain_code = self.captured_chain_code if self.captured_chain_code is not None else []
        palm_angles_left = self.captured_palm_angles_left if self.captured_palm_angles_left is not None else []
        palm_angles_right = self.captured_palm_angles_right if self.captured_palm_angles_right is not None else []
        trigger_distance_left = self.captured_trigger_distance_left if self.captured_trigger_distance_left is not None else []
        trigger_distance_right = self.captured_trigger_distance_right if self.captured_trigger_distance_right is not None else []

        # Add to dataset with separated left/right points
        sample_id = self.dataset.add_sample(
            label=label,
            points_left=points_left if points_left else None,
            points_right=points_right if points_right else None,
            hit_order=hit_order,
            chain_code=chain_code,
            palm_angles_left=palm_angles_left,
            palm_angles_right=palm_angles_right,
            trigger_distance_left=trigger_distance_left,
            trigger_distance_right=trigger_distance_right
        )
        
        # Debug output
        print(f"Sample saved: ID={sample_id}, Label={label}, Points_L={len(points_left)}, Points_R={len(points_right)}, Hit Order Length={len(hit_order)}")
        
        # Update UI
        total_points = len(points_left) + len(points_right)
        self.ui.add_sample_to_table(
            sample_id,
            label,
            "left/right",
            total_points
        )
        self.ui.update_sample_count(self.dataset.get_sample_count())
        
        # Save to file automatically
        self._auto_save_sample()
        
        # Reset captured data
        self.captured_landmarks = None
        self.captured_frame = None
        self.captured_hit_order = None
        self.captured_chain_code = None
        self.captured_palm_angles_left = None
        self.captured_palm_angles_right = None
        self.captured_trigger_distance_left = None
        self.captured_trigger_distance_right = None
        
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
        self.captured_hit_order = None
        self.last_known_landmarks = None
        self.ui.update_sample_count(0)
    
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
                show_hits=self.is_capturing  # Show hit highlights only during capture
            )
            
            # Update hit count during capture
            if self.is_capturing:
                hit_grid = self.capture.get_hit_grid_vector()
                num_hit = sum(hit_grid) if hit_grid else 0
                total_grid_points = self.capture.get_num_grid_points()
                self.ui.update_hit_count(num_hit, total_grid_points)
            
            # If capturing, store the current frame and update last known landmarks
            if self.is_capturing:
                self.captured_frame = frame.copy()
                # Update last known landmarks if hands are detected in this frame
                if landmarks_list and len(landmarks_list) > 0:
                    self.last_known_landmarks = landmarks_list
            
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

