"""
Real-time ASL Sign Recognition Interface.

Mirrors the existing dataset collection UI but performs live sign
classification using a trained ML model (loaded from .pkl files).

Workflow:
1. Initialize webcam and MediaPipe hands via existing HandCapture.
2. For each frame:
   - Detect hand landmarks.
   - Normalize to "Standard Hand" format (same as training pipeline).
   - Flatten to 63-dim vector, scale with saved scaler, predict label.
   - Update UI with detected sign and confidence (if available).
3. Provide start/stop controls and safe resource cleanup.

Assumptions:
- Model files exist at: models/asl_svm_model.pkl, models/scaler.pkl,
  models/label_encoder.pkl.
- Normalization matches training: normalize_hand_data (wrist-centered,
  3D scale, rotation-aligned).
"""

import tkinter as tk
from collections import Counter, deque
from pathlib import Path
from typing import List, Optional, Tuple

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
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 8  # set to 0 to disable majority-vote smoothing
MODEL_PATH = Path("models/asl_svm_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")


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

        # Smoothing buffer
        self.pred_buffer = deque(maxlen=SMOOTHING_WINDOW) if SMOOTHING_WINDOW > 0 else None

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
        """Load model, scaler, and label encoder."""
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

                # Update grid tracking info
                if self.grid_tracking_active:
                    self._update_grid_info()

                # Then do prediction (may be slower, but won't block display)
                prediction, confidence = self._predict_from_landmarks(landmarks_list)
                self._update_prediction_display(prediction, confidence)
        except Exception as e:
            # Log error but don't crash - keep loop running
            print(f"Error in update loop: {e}")

        self.root.after(30, self._update_loop)

    def _predict_from_landmarks(self, landmarks_list: List[dict]) -> Tuple[str, Optional[float]]:
        """Predict label from detected landmarks (first hand)."""
        # Check if model is loaded
        if self.model is None:
            return "Model not loaded", None
        
        if not landmarks_list:
            if self.pred_buffer:
                self.pred_buffer.clear()
            return "No hand detected", None

        try:
            # Use first detected hand
            hand = landmarks_list[0]
            landmarks = hand.get("landmarks", [])
            if len(landmarks) != 21:
                return "No hand detected", None

            # Normalize using same training pipeline
            normalized = normalize_hand_data(landmarks, apply_rotation=True)
            if not normalized or len(normalized) != 21:
                return "No hand detected", None

            # Flatten to 63-d vector
            flat = np.array(normalized, dtype=np.float32).flatten()
            if len(flat) != 63:
                print(f"Warning: Expected 63 features, got {len(flat)}")
                return "Feature size mismatch", None
            
            features = flat.reshape(1, -1)

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
