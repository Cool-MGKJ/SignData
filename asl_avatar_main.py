"""
Main Application for ASL Avatar Dataset Creation Tool

Standalone tool for capturing ASL gesture motion data.
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from asl_avatar_capture import ASLAvatarCapture
from asl_avatar_export import export_gesture, validate_sequence


class ASLAvatarApp:
    """Main application window for ASL avatar dataset creation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Avatar Dataset Creator")
        self.root.geometry("1200x800")
        
        # Initialize capture system
        self.capture = ASLAvatarCapture(camera_index=0, fps=30)
        self.capture.cap = cv2.VideoCapture(0)
        
        # Output directory
        self.output_dir = "asl_avatar_dataset"
        
        # UI setup
        self.setup_ui()
        
        # Start video loop
        self.update_video()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Output directory
        ttk.Label(control_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar(value=self.output_dir)
        ttk.Entry(control_frame, textvariable=self.output_dir_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2, padx=5)
        
        # Capture controls
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Capture", command=self.start_capture, width=20)
        self.start_btn.grid(row=2, column=0, columnspan=3, pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Capture", command=self.stop_capture, width=20, state=tk.DISABLED)
        self.stop_btn.grid(row=3, column=0, columnspan=3, pady=5)
        
        # Label input
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(control_frame, text="Gesture Label:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.label_var = tk.StringVar()
        self.label_entry = ttk.Entry(control_frame, textvariable=self.label_var, width=30)
        self.label_entry.grid(row=5, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(control_frame, text="(lowercase, snake_case)", font=("TkDefaultFont", 8)).grid(row=6, column=1, columnspan=2, sticky=tk.W)
        
        # Save button
        self.save_btn = ttk.Button(control_frame, text="Save Gesture", command=self.save_gesture, width=20, state=tk.DISABLED)
        self.save_btn.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Clear button
        self.clear_btn = ttk.Button(control_frame, text="Clear Capture", command=self.clear_capture, width=20, state=tk.DISABLED)
        self.clear_btn.grid(row=8, column=0, columnspan=3, pady=5)
        
        # Status
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, text="Status:").grid(row=10, column=0, sticky=tk.W, pady=5)
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=10, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        # Frame count
        self.frame_count_var = tk.StringVar(value="Frames: 0")
        ttk.Label(control_frame, textvariable=self.frame_count_var).grid(row=11, column=0, columnspan=3, pady=5)
        
        # Right panel - Video display
        video_frame = ttk.LabelFrame(main_frame, text="Camera View", padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(initialdir=self.output_dir)
        if directory:
            self.output_dir_var.set(directory)
            self.output_dir = directory
    
    def start_capture(self):
        """Start capturing frames."""
        self.capture.start_capture()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.DISABLED)
        self.status_var.set("Capturing...")
    
    def stop_capture(self):
        """Stop capturing frames."""
        self.capture.stop_capture()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.NORMAL)
        self.clear_btn.config(state=tk.NORMAL)
        
        frame_count = len(self.capture.captured_frames)
        self.status_var.set(f"Capture stopped. {frame_count} frames captured.")
        
        if frame_count == 0:
            messagebox.showwarning("No Frames", "No frames were captured. Please try again.")
    
    def save_gesture(self):
        """Save the captured gesture sequence."""
        label = self.label_var.get().strip()
        
        if not label:
            messagebox.showerror("Error", "Please enter a gesture label.")
            return
        
        # Validate label format
        label_clean = label.lower().replace(" ", "_").replace("-", "_")
        if not label_clean.replace("_", "").isalnum():
            messagebox.showerror("Error", "Label must be lowercase alphanumeric with underscores (snake_case).")
            return
        
        # Get sequence
        sequence = self.capture.get_sequence(label_clean)
        
        # Validate sequence
        is_valid, error_msg = validate_sequence(sequence)
        if not is_valid:
            messagebox.showerror("Error", f"Cannot save gesture: {error_msg}")
            return
        
        # Update output directory
        self.output_dir = self.output_dir_var.get()
        
        try:
            # Export
            filepath = export_gesture(sequence, self.output_dir)
            self.status_var.set(f"Saved: {os.path.basename(filepath)} ({sequence.num_frames} frames)")
            messagebox.showinfo("Success", f"Gesture saved to:\n{filepath}")
            
            # Clear capture
            self.clear_capture()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save gesture:\n{str(e)}")
    
    def clear_capture(self):
        """Clear captured frames."""
        self.capture.clear_capture()
        self.label_var.set("")
        self.save_btn.config(state=tk.DISABLED)
        self.clear_btn.config(state=tk.DISABLED)
        self.status_var.set("Ready")
        self.frame_count_var.set("Frames: 0")
    
    def update_video(self):
        """Update video display."""
        ret, frame = self.capture.cap.read()
        
        if ret:
            # Process frame for capture
            self.capture.capture_frame(frame)
            
            # Draw overlays
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe for visualization
            pose_results = self.capture.pose.process(frame_rgb)
            hands_results = self.capture.hands.process(frame_rgb)
            face_results = self.capture.face_mesh.process(frame_rgb)
            
            # Draw pose
            if pose_results.pose_landmarks:
                self.capture.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.capture.mp_pose.POSE_CONNECTIONS
                )
            
            # Draw hands
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.capture.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.capture.mp_hands.HAND_CONNECTIONS
                    )
            
            # Draw face mesh
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    self.capture.mp_drawing.draw_landmarks(
                        frame, face_landmarks, self.capture.mp_face_mesh.FACEMESH_CONTOURS
                    )
            
            # Add capture status overlay
            if self.capture.is_capturing:
                cv2.putText(frame, "RECORDING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame_count = len(self.capture.captured_frames)
                cv2.putText(frame, f"Frames: {frame_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (800, 600))
            
            from PIL import Image, ImageTk
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            
            # Update frame count
            if self.capture.is_capturing:
                self.frame_count_var.set(f"Frames: {len(self.capture.captured_frames)}")
        
        # Schedule next update
        self.root.after(33, self.update_video)  # ~30 FPS
    
    def on_closing(self):
        """Handle window close event."""
        self.capture.release()
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = ASLAvatarApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
