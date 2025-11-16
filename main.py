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
        
        # State
        self.is_capturing = False
        self.captured_frame = None
        self.captured_landmarks = None
        
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
    
    def stop_capture(self):
        """Stop capture and process the current frame."""
        if self.captured_frame is None:
            messagebox.showwarning("Warning", "No frame captured. Please try again.")
            return
        
        # Extract landmarks from the captured frame
        landmarks_list = self.capture.get_landmarks(self.captured_frame)
        
        if not landmarks_list:
            messagebox.showwarning(
                "Warning",
                "No hands detected in the captured frame. Please ensure your hands are visible."
            )
            return
        
        # Store for saving
        self.captured_landmarks = landmarks_list
        
        # Normalize landmarks
        hand_info = get_hand_info(landmarks_list)
        normalized_points = normalize_multiple_hands(
            landmarks_list,
            self.num_points_per_hand,
            self.spacing_ratio
        )
        
        if normalized_points is None:
            messagebox.showerror("Error", "Failed to normalize landmarks.")
            return
        
        # Display in UI
        self.ui.display_recent_sample(normalized_points, hand_info)
    
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
        
        # Normalize landmarks
        hand_info = get_hand_info(self.captured_landmarks)
        normalized_points = normalize_multiple_hands(
            self.captured_landmarks,
            self.num_points_per_hand,
            self.spacing_ratio
        )
        
        if normalized_points is None:
            return False
        
        # Add to dataset
        sample_id = self.dataset.add_sample(
            label=label,
            normalized_points=normalized_points,
            hand=hand_info
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
        self.ui.update_sample_count(0)
    
    def update_video(self):
        """Update the video feed (called repeatedly)."""
        frame = self.capture.read_frame()
        
        if frame is not None:
            # Process frame for display (with landmarks drawn)
            annotated_frame, landmarks_list = self.capture.process_frame(frame)
            
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

