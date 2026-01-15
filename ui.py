"""
User interface module using tkinter.

This module provides a desktop GUI for the ASL dataset collection tool.

Developer Notes:
- `ASLDataCollectionUI` exposes callback setters for `on_start_capture`,
  `on_stop_capture`, `on_save_sample`, `on_export_dataset`, and `on_clear_session`.
- The UI handles drawing camera frames, showing hit counts, and displaying the
  recent sample and collected samples table.
- Keep UI logic separate from data processing: UI should only call into
  `HandCapture`/`DatasetManager` via the callbacks set in `main.py`.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import cv2
from typing import Optional, Callable


class ASLDataCollectionUI:
    """Main UI class for ASL dataset collection."""
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the UI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("ASL Dataset Collection Tool")
        self.root.geometry("1200x800")
        
        # State variables
        self.is_capturing = False
        self.current_frame = None
        self.current_landmarks = None
        self.current_normalized_points = None
        self.current_hand_info = "none"
        
        # Callbacks (set by main.py)
        self.on_start_capture = None
        self.on_stop_capture = None
        self.on_save_sample = None
        self.on_export_dataset = None
        self.on_clear_session = None
        self.on_frame_update = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Create and layout all UI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left side: Camera preview and controls
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera preview
        preview_label = ttk.Label(left_frame, text="Camera Preview", font=("Arial", 12, "bold"))
        preview_label.grid(row=0, column=0, pady=(0, 5))
        
        self.camera_canvas = tk.Canvas(left_frame, width=640, height=480, bg="black")
        self.camera_canvas.grid(row=1, column=0, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(
            left_frame,
            text="Status: Ready",
            font=("Arial", 10)
        )
        self.status_label.grid(row=2, column=0, pady=(0, 10))
        
        # Control buttons frame
        controls_frame = ttk.Frame(left_frame)
        controls_frame.grid(row=3, column=0, pady=(0, 10))
        
        self.start_button = ttk.Button(
            controls_frame,
            text="Start Capture",
            command=self._handle_start_capture,
            state=tk.NORMAL
        )
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(
            controls_frame,
            text="Stop Capture",
            command=self._handle_stop_capture,
            state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Hit count display
        self.hit_count_label = ttk.Label(
            left_frame,
            text="Hit Count: 0/160",
            font=("Arial", 9)
        )
        self.hit_count_label.grid(row=4, column=0, pady=(0, 10))
        
        # Label input frame
        label_frame = ttk.LabelFrame(left_frame, text="Sign Label", padding="10")
        label_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.label_entry = ttk.Entry(label_frame, width=30, font=("Arial", 11))
        self.label_entry.grid(row=0, column=0, padx=(0, 5))
        self.label_entry.bind('<Return>', lambda e: self._handle_save_sample())
        
        self.save_button = ttk.Button(
            label_frame,
            text="Save Sample",
            command=self._handle_save_sample,
            state=tk.DISABLED
        )
        self.save_button.grid(row=0, column=1)
        
        # Right side: Sample display and dataset table
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Recent sample display
        recent_frame = ttk.LabelFrame(right_frame, text="Recent Sample", padding="10")
        recent_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        
        self.recent_sample_text = scrolledtext.ScrolledText(
            recent_frame,
            width=50,
            height=15,
            font=("Courier", 9),
            wrap=tk.WORD
        )
        self.recent_sample_text.pack(fill=tk.BOTH, expand=True)
        self.recent_sample_text.insert(tk.END, "No sample captured yet.\n")
        self.recent_sample_text.config(state=tk.DISABLED)
        
        # Dataset table
        table_frame = ttk.LabelFrame(right_frame, text="Collected Samples", padding="10")
        table_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        right_frame.rowconfigure(1, weight=1)
        
        # Create treeview for table
        columns = ("ID", "Label", "Hand", "Points")
        self.samples_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.samples_tree.heading(col, text=col)
            self.samples_tree.column(col, width=100)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.samples_tree.yview)
        self.samples_tree.configure(yscrollcommand=scrollbar.set)
        
        self.samples_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Export and Clear buttons
        action_frame = ttk.Frame(right_frame)
        action_frame.grid(row=2, column=0, pady=(0, 10))
        
        self.export_button = ttk.Button(
            action_frame,
            text="Export Dataset",
            command=self._handle_export_dataset
        )
        self.export_button.grid(row=0, column=0, padx=5)
        
        self.clear_button = ttk.Button(
            action_frame,
            text="Clear Session",
            command=self._handle_clear_session
        )
        self.clear_button.grid(row=0, column=1, padx=5)
    
    def _handle_start_capture(self):
        """Handle Start Capture button click."""
        if self.on_start_capture:
            self.on_start_capture()
        self.is_capturing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Capturing...", foreground="green")
    
    def _handle_stop_capture(self):
        """Handle Stop Capture button click."""
        if self.on_stop_capture:
            self.on_stop_capture()
        self.is_capturing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Sample captured. Enter label and save.", foreground="blue")
        self.save_button.config(state=tk.NORMAL)
        self.label_entry.focus()
    
    def _handle_save_sample(self):
        """Handle Save Sample button click."""
        label = self.label_entry.get().strip()
        if not label:
            messagebox.showwarning("Warning", "Please enter a label for the sign.")
            return
        
        if self.on_save_sample:
            success = self.on_save_sample(label)
            if success:
                self.label_entry.delete(0, tk.END)
                self.save_button.config(state=tk.DISABLED)
                self.status_label.config(text="Status: Sample saved. Ready for next capture.", foreground="black")
                messagebox.showinfo("Success", f"Sample saved with label: {label}")
            else:
                messagebox.showerror("Error", "Failed to save sample.")
    
    def _handle_export_dataset(self):
        """Handle Export Dataset button click."""
        if self.on_export_dataset:
            self.on_export_dataset()
    
    def _handle_clear_session(self):
        """Handle Clear Session button click."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all samples in this session?"):
            if self.on_clear_session:
                self.on_clear_session()
            self.samples_tree.delete(*self.samples_tree.get_children())
            self.recent_sample_text.config(state=tk.NORMAL)
            self.recent_sample_text.delete(1.0, tk.END)
            self.recent_sample_text.insert(tk.END, "No sample captured yet.\n")
            self.recent_sample_text.config(state=tk.DISABLED)
            messagebox.showinfo("Success", "Session cleared.")
    
    def update_camera_frame(self, frame: np.ndarray):
        """
        Update the camera preview with a new frame.
        
        Args:
            frame: BGR frame from OpenCV
        """
        if frame is None:
            return
        
        # Resize if needed
        height, width = frame.shape[:2]
        if width > 640 or height > 480:
            scale = min(640 / width, 480 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update canvas
        self.camera_canvas.delete("all")
        self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.camera_canvas.image = photo  # Keep a reference
    
    def display_recent_sample(
        self,
        normalized_points: list,
        hand_info: str,
        label: Optional[str] = None
    ):
        """
        Display the most recently captured sample.
        
        Args:
            normalized_points: List of normalized (x, y, z) tuples
            hand_info: String describing which hand(s) detected
            label: Optional label for the sample
        """
        try:
            self.recent_sample_text.config(state=tk.NORMAL)
            self.recent_sample_text.delete(1.0, tk.END)
            
            # Check if normalized_points is valid
            if not normalized_points:
                output = "No points available to display.\n"
                self.recent_sample_text.insert(tk.END, output)
                self.recent_sample_text.config(state=tk.DISABLED)
                return
            
            # Format output
            output = f"Hand: {hand_info}\n"
            if label:
                output += f"Label: {label}\n"
            output += f"Number of points: {len(normalized_points)}\n\n"
            output += "Normalized 3D Points (Numbered):\n"
            output += "-" * 70 + "\n"
            output += f"{'No.':<4} {'Landmark':<12} {'X':<12} {'Y':<12} {'Z':<12}\n"
            output += "-" * 70 + "\n"

            # MediaPipe hand landmark names for reference
            landmark_names = {
                0: "wrist", 1: "thumb_cmc", 2: "thumb_mcp", 3: "thumb_ip", 4: "thumb_tip",
                5: "index_mcp", 6: "index_pip", 7: "index_dip", 8: "index_tip",
                9: "middle_mcp", 10: "middle_pip", 11: "middle_dip", 12: "middle_tip",
                13: "ring_mcp", 14: "ring_pip", 15: "ring_dip", 16: "ring_tip",
                17: "pinky_mcp", 18: "pinky_pip", 19: "pinky_dip", 20: "pinky_tip"
            }

            # Handle both tuple format and other formats
            for i, point in enumerate(normalized_points):
                # Handle different point formats
                if isinstance(point, (list, tuple)) and len(point) >= 3:
                    x, y, z = point[0], point[1], point[2]
                elif isinstance(point, dict):
                    x = point.get('x', 0.0)
                    y = point.get('y', 0.0)
                    z = point.get('z', 0.0)
                else:
                    print(f"Warning: Unexpected point format at index {i}: {point}")
                    continue
                
                landmark_name = landmark_names.get(i, f"point_{i}")
                output += f"{i:2d}. {landmark_name:<12} {x:<12.6f} {y:<12.6f} {z:<12.6f}\n"
            
            self.recent_sample_text.insert(tk.END, output)
            self.recent_sample_text.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error displaying recent sample: {e}")
            import traceback
            traceback.print_exc()
            self.recent_sample_text.config(state=tk.NORMAL)
            self.recent_sample_text.delete(1.0, tk.END)
            self.recent_sample_text.insert(tk.END, f"Error displaying sample: {str(e)}\n")
            self.recent_sample_text.config(state=tk.DISABLED)
    
    def add_sample_to_table(self, sample_id: int, label: str, hand: str, num_points: int):
        """
        Add a sample to the dataset table.
        
        Args:
            sample_id: Unique sample ID
            label: Sign label
            hand: Hand information
            num_points: Number of points in the sample
        """
        self.samples_tree.insert("", tk.END, values=(sample_id, label, hand, num_points))
    
    def update_sample_count(self, count: int):
        """
        Update the status with current sample count.
        
        Args:
            count: Number of samples collected
        """
        current_status = self.status_label.cget("text")
        if "Ready" in current_status or "Sample saved" in current_status:
            self.status_label.config(text=f"Status: {count} samples collected. Ready for next capture.")
    
    def update_hit_count(self, num_hit: int, total: int):
        """
        Update the hit count display.
        
        Args:
            num_hit: Number of voxels hit
            total: Total number of voxels
        """
        self.hit_count_label.config(text=f"Hit Count: {num_hit}/{total}")
    
    def set_callbacks(
        self,
        on_start_capture: Callable,
        on_stop_capture: Callable,
        on_save_sample: Callable,
        on_export_dataset: Callable,
        on_clear_session: Callable
    ):
        """
        Set callback functions for UI events.
        
        Args:
            on_start_capture: Called when Start Capture is clicked
            on_stop_capture: Called when Stop Capture is clicked
            on_save_sample: Called when Save Sample is clicked (takes label, returns bool)
            on_export_dataset: Called when Export Dataset is clicked
            on_clear_session: Called when Clear Session is confirmed
        """
        self.on_start_capture = on_start_capture
        self.on_stop_capture = on_stop_capture
        self.on_save_sample = on_save_sample
        self.on_export_dataset = on_export_dataset
        self.on_clear_session = on_clear_session

