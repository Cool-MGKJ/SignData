"""
MiDaS monocular depth estimation module.

This module provides depth estimation using MiDaS models without requiring
a hardware depth camera. It uses monocular depth estimation from RGB frames.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple
import warnings

# Suppress warnings about model loading
warnings.filterwarnings("ignore", category=UserWarning)


class MiDaSDepthProvider:
    """
    Provides monocular depth estimation using MiDaS models.
    
    This class loads a MiDaS depth estimation model and processes RGB frames
    to produce relative depth maps. The depth values are normalized to 0.0-1.0.
    """
    
    def __init__(self, model_type: str = "DPT_Large"):
        """
        Initialize the MiDaS depth provider.
        
        Args:
            model_type: Type of MiDaS model to use. Options:
                       - "DPT_Large" (default, most accurate)
                       - "DPT_Hybrid"
                       - "MiDaS_small" (fastest, least accurate)
        """
        self.model_type = model_type
        self.model = None
        self.device = None
        self.transform = None
        self.current_depth_map = None
        self.is_initialized = False
        
    def start(self) -> bool:
        """
        Initialize and load the MiDaS model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check if torch is available and set device
            try:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
            except:
                self.device = torch.device("cpu")
            
            # Try to load MiDaS model
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                
                # Map model types to HuggingFace model names
                model_map = {
                    "DPT_Large": "Intel/dpt-large",
                    "DPT_Hybrid": "Intel/dpt-hybrid",
                    "MiDaS_small": "Intel/dpt-depth-estimation"
                }
                
                model_name = model_map.get(self.model_type, "Intel/dpt-large")
                
                print(f"Loading MiDaS model: {model_name}...")
                self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                
                self.transform = AutoImageProcessor.from_pretrained(model_name)
                
                self.is_initialized = True
                print(f"MiDaS model loaded successfully on {self.device}")
                return True
                
            except ImportError:
                # Fallback: try using torch.hub to load MiDaS
                print("transformers not available, trying torch.hub...")
                try:
                    # Load MiDaS from torch hub
                    self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
                    self.model.to(self.device)
                    self.model.eval()
                    
                    # Create a simple transform
                    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                    if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
                        self.transform = midas_transforms.dpt_transform
                    else:
                        self.transform = midas_transforms.small_transform
                    
                    self.is_initialized = True
                    print(f"MiDaS model loaded successfully from torch.hub on {self.device}")
                    return True
                except Exception as e:
                    print(f"Failed to load MiDaS model: {e}")
                    return False
                    
        except Exception as e:
            print(f"Error initializing MiDaS depth provider: {e}")
            self.is_initialized = False
            return False
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Process a frame to compute depth map.
        
        Args:
            frame: BGR frame from OpenCV (numpy array)
            
        Returns:
            True if processing successful, False otherwise
        """
        if not self.is_initialized or self.model is None:
            return False
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transform and prepare input
            if hasattr(self.transform, 'preprocess'):
                # HuggingFace AutoImageProcessor
                inputs = self.transform(rgb_frame, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # torch.hub transform
                input_batch = self.transform(rgb_frame).to(self.device)
                inputs = {"pixel_values": input_batch}
            
            # Run inference
            with torch.no_grad():
                if "pixel_values" in inputs:
                    outputs = self.model(inputs["pixel_values"])
                else:
                    outputs = self.model(**inputs)
                
                # Extract depth prediction
                if hasattr(outputs, 'predicted_depth'):
                    prediction = outputs.predicted_depth
                elif isinstance(outputs, dict) and 'predicted_depth' in outputs:
                    prediction = outputs['predicted_depth']
                else:
                    # Fallback: assume outputs is the tensor directly
                    prediction = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                
                # Convert to numpy
                depth = prediction.cpu().numpy()
                
                # Resize to original frame size if needed
                if depth.ndim == 4:
                    depth = depth.squeeze()
                if depth.ndim == 3:
                    depth = depth[0] if depth.shape[0] == 1 else depth
                
                h, w = frame.shape[:2]
                if depth.shape != (h, w):
                    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # Normalize depth to 0.0-1.0 range
                depth_min = np.min(depth)
                depth_max = np.max(depth)
                if depth_max > depth_min:
                    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = np.zeros_like(depth)
                
                self.current_depth_map = depth_normalized.astype(np.float32)
                return True
                
        except Exception as e:
            print(f"Error processing frame for depth: {e}")
            self.current_depth_map = None
            return False
    
    def get_depth_map(self) -> Optional[np.ndarray]:
        """
        Get the current depth map.
        
        Returns:
            2D numpy array of depth values (0.0-1.0), or None if not available
        """
        return self.current_depth_map
    
    def get_depth_at(self, u: int, v: int) -> float:
        """
        Get depth value at a specific pixel location.
        
        Args:
            u: X pixel coordinate (column)
            v: Y pixel coordinate (row)
            
        Returns:
            Depth value (0.0-1.0), or 0.0 if not available or out of bounds
        """
        if self.current_depth_map is None:
            return 0.0
        
        h, w = self.current_depth_map.shape
        if 0 <= v < h and 0 <= u < w:
            return float(self.current_depth_map[v, u])
        return 0.0
    
    def get_depth_at_normalized(self, x: float, y: float, frame_width: int, frame_height: int) -> float:
        """
        Get depth value at normalized coordinates (0.0-1.0).
        
        Args:
            x: Normalized x coordinate (0.0-1.0)
            y: Normalized y coordinate (0.0-1.0)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Depth value (0.0-1.0), or 0.0 if not available
        """
        u = int(x * frame_width)
        v = int(y * frame_height)
        return self.get_depth_at(u, v)
    
    def is_active(self) -> bool:
        """Check if the depth provider is active and ready."""
        return self.is_initialized and self.model is not None
    
    def release(self):
        """Release resources."""
        if self.model is not None:
            del self.model
            self.model = None
        self.current_depth_map = None
        self.is_initialized = False

