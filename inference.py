"""
Real-time ASL Sign Recognition Interface.

Mirrors the existing dataset collection UI but performs live sign
classification using a trained ML model (loaded from .pkl files).

Workflow:
1. Initialize webcam and MediaPipe hands via existing HandCapture.
2. For each frame:
   - Detect hand landmarks (uses only first detected hand).
   - Normalize to "Standard Hand" format (same as training pipeline).
   - Extract features: 63-dim points + hit_order + chain_code + palm_angles.
   - Scale with saved scaler, predict label.
   - Update UI with detected sign and confidence (if available).
3. Provide start/stop controls and safe resource cleanup.

Assumptions:
- Model files exist at: models/asl_svm_model.pkl, models/scaler.pkl,
  models/label_encoder.pkl, models/feature_config.pkl.
- Normalization matches training: normalize_hand_data (wrist-centered,
  3D scale, rotation-aligned).
- Uses single hand (63 features), excludes trigger_distance.
"""

import tkinter as tk
from collections import Counter, deque
from pathlib import Path
from typing import List, Optional, Tuple
from time import time

import cv2
import joblib
import numpy as np
from tkinter import messagebox

from capture import HandCapture
from normalization import normalize_hand_data


# UI configuration
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CAMERA_PREVIEW_WIDTH = 640
CAMERA_PREVIEW_HEIGHT = 480
PREDICTION_FONT = ("Arial", 32, "bold")
STATUS_FONT = ("Arial", 12, "bold")

# Inference configuration
CONFIDENCE_THRESHOLD = 0.5  # Detection fires when confidence >= 0.5
SMOOTHING_WINDOW = 8  # set to 0 to disable majority-vote smoothing
RESET_DELAY = 1.0  # Seconds to wait after first sign detection before reset
MODEL_PATH = Path("models/asl_svm_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")
FEATURE_CONFIG_PATH = Path("models/feature_config.pkl")


class ASLInferenceApp:
    """Real-time ASL sign inference application."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("ASL Sign Recognition")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        # UI elements
        self.camera_canvas = None
        self.status_label = None
        self.prediction_label = None
        self.confidence_label = None
        self.start_button = None
        self.stop_button = None
        self.hit_count_label = None
        self.chain_code_label = None
        self.voxel_path_label = None

        # Capture / inference state
        self.capture = None
        self.is_running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.grid_tracking_active = False

        # Model artifacts
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.max_hit_order_len = 100  # Default, will be loaded from config
        self.max_chain_code_len = 100  # Default, will be loaded from config
        self.max_palm_angles_len = 200  # Default, will be loaded from config

        # Smoothing buffer
        self.pred_buffer = deque(maxlen=SMOOTHING_WINDOW) if SMOOTHING_WINDOW > 0 else None
        
        # Auto-reset tracking
        self.first_detection_time: Optional[float] = None

        self._build_ui()
        self._load_models()

    def _build_ui(self) -> None:
        """Create and layout UI components."""
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Camera preview
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=(0, 10), fill=tk.Y)

        tk.Label(left_frame, text="Camera Preview", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        self.camera_canvas = tk.Canvas(
            left_frame,
            width=CAMERA_PREVIEW_WIDTH,
            height=CAMERA_PREVIEW_HEIGHT,
            bg="black",
        )
        self.camera_canvas.pack()

        # Status
        self.status_label = tk.Label(left_frame, text="Status: Camera OFF", font=STATUS_FONT, fg="red")
        self.status_label.pack(pady=(10, 0))

        # Hit count display (160-point two-layer grid)
        self.hit_count_label = tk.Label(
            left_frame,
            text="Hit Count: 0/160",
            font=("Arial", 9)
        )
        self.hit_count_label.pack(pady=(5, 0))

        # Controls
        controls = tk.Frame(left_frame)
        controls.pack(pady=(10, 0))
        self.start_button = tk.Button(controls, text="Start Inference", command=self.start)
        self.start_button.grid(row=0, column=0, padx=5)
        self.stop_button = tk.Button(controls, text="Stop", state=tk.DISABLED, command=self.stop)
        self.stop_button.grid(row=0, column=1, padx=5)

        # Right: Prediction display
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right_frame, text="Detected Sign", font=("Arial", 16, "bold")).pack(pady=(0, 10))
        self.prediction_label = tk.Label(right_frame, text="–", font=PREDICTION_FONT, fg="blue")
        self.prediction_label.pack(pady=(0, 10))

        self.confidence_label = tk.Label(right_frame, text="Confidence: –", font=("Arial", 14))
        self.confidence_label.pack(pady=(0, 20))

        # Voxel path and chain code display
        info_frame = tk.LabelFrame(right_frame, text="Grid Tracking Info", font=("Arial", 12, "bold"))
        info_frame.pack(pady=(10, 0), padx=10, fill=tk.BOTH, expand=True)

        # Voxel path display
        tk.Label(info_frame, text="Voxel Path (Hit Order):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.voxel_path_label = tk.Label(
            info_frame,
            text="[]",
            font=("Courier", 9),
            justify=tk.LEFT,
            wraplength=400
        )
        self.voxel_path_label.pack(anchor=tk.W, padx=10, pady=(0, 10))

        # Chain code display
        tk.Label(info_frame, text="Chain Code:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.chain_code_label = tk.Label(
            info_frame,
            text="[]",
            font=("Courier", 9),
            justify=tk.LEFT,
            wraplength=400
        )
        self.chain_code_label.pack(anchor=tk.W, padx=10, pady=(0, 10))

    def _load_models(self) -> None:
        """Load model, scaler, label encoder, and feature config."""
        try:
            print(f"Loading model from: {MODEL_PATH}")
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            self.model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully. Type: {type(self.model)}")
            
            print(f"Loading scaler from: {SCALER_PATH}")
            if not SCALER_PATH.exists():
                raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
            self.scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded successfully. Type: {type(self.scaler)}")
            
            print(f"Loading label encoder from: {ENCODER_PATH}")
            if not ENCODER_PATH.exists():
                raise FileNotFoundError(f"Label encoder file not found: {ENCODER_PATH}")
            self.label_encoder = joblib.load(ENCODER_PATH)
            print(f"Label encoder loaded successfully. Classes: {self.label_encoder.classes_}")
            
            # Load feature configuration if available
            if FEATURE_CONFIG_PATH.exists():
                feature_config = joblib.load(FEATURE_CONFIG_PATH)
                self.max_hit_order_len = feature_config.get("max_hit_order_len", 100)
                self.max_chain_code_len = feature_config.get("max_chain_code_len", 100)
                self.max_palm_angles_len = feature_config.get("max_palm_angles_len", 200)
                print(f"Feature config loaded:")
                print(f"  - hit_order_len: {self.max_hit_order_len}")
                print(f"  - chain_code_len: {self.max_chain_code_len}")
                print(f"  - palm_angles_len: {self.max_palm_angles_len}")
            else:
                print("Warning: Feature config not found, using defaults")
            
        except FileNotFoundError as exc:
            error_msg = f"Model files missing: {exc}"
            print(error_msg)
            messagebox.showerror("Model files missing", error_msg)
            self.root.after(100, self.root.destroy)
        except Exception as exc:
            error_msg = f"Failed to load models: {exc}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            messagebox.showerror("Model load error", error_msg)
            self.root.after(100, self.root.destroy)

    def start(self) -> None:
        """Start webcam capture and inference loop."""
        if self.is_running:
            return

        self.capture = HandCapture()
        if not self.capture.initialize():
            messagebox.showerror("Error", "Failed to initialize camera.")
            self.capture = None
            return

        # Start grid tracking for voxel path and chain code
        self.capture.start_grid_tracking()
        self.grid_tracking_active = True

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Camera ON", fg="green")
        self._update_loop()

    def stop(self) -> None:
        """Stop inference and release resources."""
        self.is_running = False
        if self.capture and self.grid_tracking_active:
            self.capture.stop_grid_tracking()
            self.grid_tracking_active = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Camera OFF", fg="red")
        # Reset displays
        self.hit_count_label.config(text="Hit Count: 0/160")
        self.voxel_path_label.config(text="[]")
        self.chain_code_label.config(text="[]")
        # Reset auto-reset tracking
        self.first_detection_time = None
        if self.capture:
            self.capture.release()
            self.capture = None

    def _update_loop(self) -> None:
        """Main loop: capture frame, infer, update UI."""
        if not self.is_running or not self.capture:
            return

        try:
            frame = self.capture.read_frame()
            if frame is not None:
                # Process frame with grid visualization enabled
                annotated, landmarks_list = self.capture.process_frame(
                    frame,
                    track_grid_hits=self.grid_tracking_active,
                    draw_grid=True,  # Always show the 160-point two-layer grid
                    show_hits=True,  # Show hit highlights
                )

                # Update camera display first (non-blocking)
                self._update_camera_canvas(annotated)

                # Get hit_order, chain_code, and palm_angles if available
                hit_order = []
                chain_code = []
                palm_angles_left = []
                palm_angles_right = []
                
                if self.grid_tracking_active:
                    hit_order = self.capture.get_hit_order() or []
                    chain_code = self.capture.get_chain_code() or []
                    palm_angles_left = self.capture.get_palm_angles_left() or []
                    palm_angles_right = self.capture.get_palm_angles_right() or []
                    self._update_grid_info()

                # Then do prediction (may be slower, but won't block display)
                prediction, confidence = self._predict_from_landmarks(
                    landmarks_list, hit_order, chain_code, 
                    palm_angles_left, palm_angles_right
                )
                self._update_prediction_display(prediction, confidence)
                
                # Check for auto-reset condition
                self._check_auto_reset(prediction, confidence)
        except Exception as e:
            # Log error but don't crash - keep loop running
            print(f"Error in update loop: {e}")

        self.root.after(30, self._update_loop)

    def _predict_from_landmarks(
        self, 
        landmarks_list: List[dict], 
        hit_order: List[int] = None, 
        chain_code: List[int] = None,
        palm_angles_left: List[tuple] = None,
        palm_angles_right: List[tuple] = None
    ) -> Tuple[str, Optional[float]]:
        """
        Predict label from detected landmarks and motion features.
        
        Uses only detected hand (single hand = 63 features), matching training pipeline.
        """
        # Check if model is loaded
        if self.model is None:
            return "Model not loaded", None
        
        if not landmarks_list:
            if self.pred_buffer:
                self.pred_buffer.clear()
            return "No hand detected", None

        try:
            # Use only detected hand (single hand = 63 features, matching training)
            points_flat = None
            
            # Process first detected hand only
            for hand_data in landmarks_list:
                landmarks = hand_data.get("landmarks", [])
                if len(landmarks) == 21:
                    # Normalize using same training pipeline
                    normalized = normalize_hand_data(landmarks, apply_rotation=True)
                    if normalized and len(normalized) == 21:
                        # Flatten to 63-d vector (single hand)
                        points_flat = np.array(normalized, dtype=np.float32).flatten()
                        break  # Use only first detected hand
            
            if points_flat is None or len(points_flat) != 63:
                print(f"Warning: Expected 63 features (single hand), got {len(points_flat) if points_flat is not None else 0}")
                return "Feature size mismatch", None
            
            # Pad to 63 features if needed (shouldn't happen, but safety check)
            if len(points_flat) < 63:
                points_flat = np.pad(points_flat, (0, 63 - len(points_flat)), mode='constant', constant_values=0.0)
            elif len(points_flat) > 63:
                points_flat = points_flat[:63]
            
            # Extract features: points_flat + hit_order + chain_code + palm_angles (NO trigger_distance)
            features = self._extract_features(
                points_flat,
                hit_order or [],
                chain_code or [],
                palm_angles_left or [],
                palm_angles_right or []
            )

            # Scale features if scaler available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            else:
                print("Warning: Scaler not loaded")

            # Predict label
            if self.model is None:
                return "Model not loaded", None
                
            proba = None
            if hasattr(self.model, "predict_proba"):
                proba_vals = self.model.predict_proba(features)
                proba = float(np.max(proba_vals))
                pred_idx = int(np.argmax(proba_vals, axis=1)[0])
            else:
                pred_idx = int(self.model.predict(features)[0])

            if self.label_encoder is not None:
                label = self.label_encoder.inverse_transform([pred_idx])[0]
            else:
                label = str(pred_idx)
                print("Warning: Label encoder not loaded")

            # Confidence gating
            if proba is not None and proba < CONFIDENCE_THRESHOLD:
                label = "Low confidence"

            # Smoothing
            if self.pred_buffer is not None and label not in ("No hand detected", "Low confidence"):
                self.pred_buffer.append(label)
                label = self._smoothed_label()

            return label, proba
        except Exception as e:
            # Log error with full traceback
            import traceback
            print(f"Prediction error: {e}")
            traceback.print_exc()
            return "Prediction error", None

    def _extract_features(
        self, 
        points_flat: np.ndarray, 
        hit_order: List[int], 
        chain_code: List[int],
        palm_angles_left: List[tuple],
        palm_angles_right: List[tuple]
    ) -> np.ndarray:
        """
        Extract features matching training pipeline: points_flat + hit_order + chain_code + palm_angles.
        
        Uses single hand (63 features) and excludes trigger_distance.
        
        Args:
            points_flat: 63-dimensional flattened hand landmarks (single hand)
            hit_order: List of voxel indices hit during tracking
            chain_code: List of chain code direction indices
            palm_angles_left: List of (yaw, pitch, roll) tuples for left hand
            palm_angles_right: List of (yaw, pitch, roll) tuples for right hand
            
        Returns:
            Combined feature vector
        """
        # Pad or truncate hit_order
        hit_order_padded = list(hit_order[:self.max_hit_order_len]) + [0] * (self.max_hit_order_len - len(hit_order))
        
        # Pad or truncate chain_code
        chain_code_padded = list(chain_code[:self.max_chain_code_len]) + [0] * (self.max_chain_code_len - len(chain_code))
        
        # Flatten palm_angles from both hands
        # Note: face_grid_3d returns (yaw, pitch, roll), but dataset stores [pitch, yaw, roll]
        # Training expects [pitch, yaw, roll] format, so we need to reorder
        palm_angles_flat = []
        if palm_angles_left:
            for angle_tuple in palm_angles_left:
                if isinstance(angle_tuple, (tuple, list)) and len(angle_tuple) == 3:
                    # Convert (yaw, pitch, roll) to [pitch, yaw, roll] to match training
                    yaw, pitch, roll = angle_tuple
                    palm_angles_flat.extend([pitch, yaw, roll])
        
        if palm_angles_right:
            for angle_tuple in palm_angles_right:
                if isinstance(angle_tuple, (tuple, list)) and len(angle_tuple) == 3:
                    # Convert (yaw, pitch, roll) to [pitch, yaw, roll] to match training
                    yaw, pitch, roll = angle_tuple
                    palm_angles_flat.extend([pitch, yaw, roll])
        
        # Pad palm_angles_flat
        palm_angles_padded = list(palm_angles_flat[:self.max_palm_angles_len]) + [0.0] * (self.max_palm_angles_len - len(palm_angles_flat))
        
        # Combine all features: points_flat + hit_order + chain_code + palm_angles (NO trigger_distance)
        combined = np.concatenate([
            points_flat,
            np.array(hit_order_padded, dtype=np.float32),
            np.array(chain_code_padded, dtype=np.float32),
            np.array(palm_angles_padded, dtype=np.float32)
        ])
        
        return combined.reshape(1, -1)

    def _smoothed_label(self) -> str:
        """Return majority label from buffer."""
        if not self.pred_buffer:
            return "No hand detected"
        counts = Counter(self.pred_buffer)
        return counts.most_common(1)[0][0]

    def _update_camera_canvas(self, frame: np.ndarray) -> None:
        """Update the Tkinter canvas with the latest frame."""
        if frame is None:
            return

        try:
            # Resize to fit preview
            h, w = frame.shape[:2]
            if w > CAMERA_PREVIEW_WIDTH or h > CAMERA_PREVIEW_HEIGHT:
                scale = min(CAMERA_PREVIEW_WIDTH / w, CAMERA_PREVIEW_HEIGHT / h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Frame is in BGR from process_frame, convert to RGB for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode as PPM format for Tkinter
            img = tk.PhotoImage(master=self.camera_canvas, data=cv2.imencode(".ppm", rgb)[1].tobytes())
            
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=img)
            # Keep reference to prevent garbage collection
            self.camera_canvas.image = img
        except Exception as e:
            # Don't crash on display errors
            print(f"Error updating camera canvas: {e}")

    def _update_grid_info(self) -> None:
        """Update grid tracking information display."""
        if not self.capture:
            return

        try:
            # Get hit order (voxel path)
            hit_order = self.capture.get_hit_order() or []
            num_hits = len(hit_order)
            
            # Get total grid points (should be 160 for 8x10x2 grid)
            total_grid_points = self.capture.get_num_grid_points()
            
            # Update hit count
            self.hit_count_label.config(text=f"Hit Count: {num_hits}/{total_grid_points}")
            
            # Update voxel path display (show first 50 for readability)
            if hit_order:
                display_path = hit_order[:50] if len(hit_order) > 50 else hit_order
                path_text = str(display_path)
                if len(hit_order) > 50:
                    path_text += f" ... (+{len(hit_order) - 50} more)"
                self.voxel_path_label.config(text=path_text)
            else:
                self.voxel_path_label.config(text="[]")
            
            # Get chain code
            chain_code = self.capture.get_chain_code() or []
            
            # Update chain code display (show first 50 for readability)
            if chain_code:
                display_chain = chain_code[:50] if len(chain_code) > 50 else chain_code
                chain_text = str(display_chain)
                if len(chain_code) > 50:
                    chain_text += f" ... (+{len(chain_code) - 50} more)"
                self.chain_code_label.config(text=chain_text)
            else:
                self.chain_code_label.config(text="[]")
        except Exception as e:
            # Don't crash on grid info errors
            print(f"Error updating grid info: {e}")

    def _check_auto_reset(self, label: str, confidence: Optional[float]) -> None:
        """
        Check if we should auto-reset after sign detection.
        
        Resets grid tracking if:
        - A valid hand sign is first detected (confidence >= CONFIDENCE_THRESHOLD)
        - One second has elapsed since that first detection
        """
        current_time = time()
        
        # Check if we have a valid sign detection (confidence >= threshold and valid label)
        if (confidence is not None and 
            confidence >= CONFIDENCE_THRESHOLD and 
            label not in ("No hand detected", "Low confidence", "Model not loaded", "Feature size mismatch", "Prediction error")):
            
            # First time detecting a sign - start timer
            if self.first_detection_time is None:
                self.first_detection_time = current_time
                print(f"Sign detected: '{label}' with {confidence:.2f} confidence - reset timer started")
            
            # Check if 1 second has elapsed since first detection
            elif self.first_detection_time is not None:
                elapsed = current_time - self.first_detection_time
                if elapsed >= RESET_DELAY:
                    # Reset grid tracking to start fresh for next sign
                    print(f"Auto-reset: 1 second elapsed after first sign detection (current: '{label}' with {confidence:.2f})")
                    self._reset_tracking()
                    self.first_detection_time = None
        else:
            # No valid sign detected or confidence dropped - reset timer
            if self.first_detection_time is not None:
                self.first_detection_time = None

    def _reset_tracking(self) -> None:
        """Reset grid tracking to start fresh detection."""
        if self.capture and self.grid_tracking_active:
            # Stop and restart grid tracking to clear all accumulated data
            self.capture.stop_grid_tracking()
            self.capture.start_grid_tracking()
            
            # Clear prediction buffer
            if self.pred_buffer:
                self.pred_buffer.clear()
            
            # Reset UI displays
            self.hit_count_label.config(text="Hit Count: 0/160")
            self.voxel_path_label.config(text="[]")
            self.chain_code_label.config(text="[]")
            
            print("Grid tracking reset - ready for next sign detection")

    def _update_prediction_display(self, label: str, confidence: Optional[float]) -> None:
        """Update UI with latest prediction."""
        self.prediction_label.config(text=label)
        if confidence is not None:
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
        else:
            self.confidence_label.config(text="Confidence: –")

    def run(self) -> None:
        """Start the Tkinter main loop."""
        try:
            self.root.mainloop()
        finally:
            self.stop()


def main() -> None:
    app = ASLInferenceApp()
    app.run()


if __name__ == "__main__":
    main()
