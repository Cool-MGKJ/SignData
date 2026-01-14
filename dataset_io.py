"""
Dataset input/output module.

This module handles storing and saving labelled ASL sign samples
in both CSV and JSON formats.

Developer Notes:
- `DatasetManager.add_sample()` stores points separately for left and right hands.
- No timestamps are included to reduce data size.
- Samples include `hit_order` and `chain_code` metadata for later clustering and analysis.
- Export helpers write JSON or CSV suitable for ingestion by ML pipelines.
"""

import json
import csv
import os
from typing import List, Tuple, Optional


class DatasetManager:
    """Manages in-memory dataset and file I/O for ASL samples."""
    
    def __init__(self):
        """Initialize an empty dataset."""
        self.samples = []  # List of sample dicts
        self.next_id = 1
    
    def add_sample(
        self,
        label: str,
        points_left: Optional[List[Tuple[float, float, float]]] = None,
        points_right: Optional[List[Tuple[float, float, float]]] = None,
        hit_order: Optional[List[int]] = None,
        chain_code: Optional[List[int]] = None,
        palm_angles_left: Optional[List[tuple]] = None,
        palm_angles_right: Optional[List[tuple]] = None,
        trigger_distance_left: Optional[List[float]] = None,
        trigger_distance_right: Optional[List[float]] = None,
        trigger_point_left: Optional[List[Tuple[float, float, float]]] = None,
        trigger_point_right: Optional[List[Tuple[float, float, float]]] = None
    ) -> int:
        """
        Add a new sample to the dataset.

        Args:
            label: The ASL sign label (e.g., "hello", "thank you")
            points_left: List of normalized (x, y, z) tuples for left hand
            points_right: List of normalized (x, y, z) tuples for right hand
            hit_order: Ordered list of voxel indices representing hit sequence
            chain_code: List of direction indices (0-25) representing trajectory
            palm_angles_left: List of (yaw, pitch, roll) tuples for left hand
            palm_angles_right: List of (yaw, pitch, roll) tuples for right hand
            trigger_distance_left: List of distances from trigger point to nose for left hand
            trigger_distance_right: List of distances from trigger point to nose for right hand
            trigger_point_left: List of (x, y, z) wrist positions for left hand
            trigger_point_right: List of (x, y, z) wrist positions for right hand

        Returns:
            The unique sample ID assigned to this sample
        """
        sample_id = self.next_id
        self.next_id += 1
        
        # Store points as numbered objects for both hands
        def format_points(points_list):
            if points_list is None:
                return []
            return [
                {'index': idx, 'x': point[0], 'y': point[1], 'z': point[2]}
                for idx, point in enumerate(points_list)
            ]
        
        left_points = format_points(points_left)
        right_points = format_points(points_right)
        
        # Format trigger points as well
        def format_trigger_points(trigger_list):
            if trigger_list is None:
                return []
            return [{'x': point[0], 'y': point[1], 'z': point[2]} for point in trigger_list]
        
        trigger_points_left = format_trigger_points(trigger_point_left)
        trigger_points_right = format_trigger_points(trigger_point_right)
        
        sample = {
            'id': sample_id,
            'label': label,
            'points_left': left_points,
            'points_right': right_points,
            'hit_order': hit_order if hit_order is not None else [],
            'chain_code': chain_code if chain_code is not None else [],
            'palm_angles_left': palm_angles_left if palm_angles_left is not None else [],
            'palm_angles_right': palm_angles_right if palm_angles_right is not None else [],
            'trigger_distance_left': trigger_distance_left if trigger_distance_left is not None else [],
            'trigger_distance_right': trigger_distance_right if trigger_distance_right is not None else [],
            'trigger_point_left': trigger_points_left,
            'trigger_point_right': trigger_points_right
        }

        self.samples.append(sample)
        print(f"Added sample to dataset: ID={sample_id}, Label={label}, Points_L={len(left_points)}, Points_R={len(right_points)}, Hit Order={len(sample['hit_order'])}, Trigger Points L={len(trigger_points_left)}, R={len(trigger_points_right)}")
        return sample_id
    
    def get_all_samples(self) -> List[dict]:
        """
        Get all samples in the current session.
        
        Returns:
            List of sample dictionaries
        """
        return self.samples.copy()
    
    def get_sample_count(self) -> int:
        """Get the number of samples in the current session."""
        return len(self.samples)
    
    def clear(self):
        """Clear all samples from the current session."""
        self.samples = []
        self.next_id = 1
    
    def save_as_csv(self, path: str, append: bool = False):
        """
        Save the dataset to a CSV file.
        
        The CSV format will be:
        - Header row: id,label,hand,num_points,point_0_x,point_0_y,point_0_z,...
        - Data rows: sample values
        
        Args:
            path: File path to save to
            append: If True, append to existing file (default: False)
        """
        if not self.samples:
            print("No samples to save.")
            return
        
        # Determine if file exists and has content
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        
        # Get the maximum number of points to determine column count
        max_points = max(s['num_points'] for s in self.samples)
        num_coords = max_points * 3  # x, y, z for each point
        
        # Determine maximum hit-order length to size columns
        max_hit_order_len = 0
        for s in self.samples:
            if 'hit_order' in s:
                max_hit_order_len = max(max_hit_order_len, len(s['hit_order']))
        
        # Create header
        header = ['id', 'label', 'hand', 'num_points']
        for i in range(max_points):
            header.extend([f'point_{i}_x', f'point_{i}_y', f'point_{i}_z'])
        
        if max_hit_order_len > 0:
            header.append('hit_order_length')
            for i in range(max_hit_order_len):
                header.append(f'hit_order_{i}')
        
        mode = 'a' if (append and file_exists) else 'w'
        newline = ''  # Required for CSV on Windows
        
        with open(path, mode, newline=newline, encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header only if creating new file or appending to empty file
            if mode == 'w' or not file_exists:
                writer.writerow(header)
            
            # Write samples
            for sample in self.samples:
                row = [
                    sample['id'],
                    sample['label'],
                    sample['hand'],
                    sample['num_points']
                ]
                
                # Handle both numbered format (new) and flattened format (old)
                points = sample.get('points', [])
                if points and isinstance(points[0], dict):
                    # New numbered format: [{"index": 0, "x": ..., "y": ..., "z": ...}, ...]
                    for i in range(max_points):
                        if i < len(points):
                            row.extend([points[i]['x'], points[i]['y'], points[i]['z']])
                        else:
                            row.extend([0.0, 0.0, 0.0])
                else:
                    # Old flattened format: [x1, y1, z1, x2, y2, z2, ...]
                    # Or use points_flat if available
                    flat_points = sample.get('points_flat', points)
                    for i in range(0, len(flat_points), 3):
                        if i + 2 < len(flat_points):
                            row.extend([flat_points[i], flat_points[i+1], flat_points[i+2]])
                        else:
                            row.extend([0.0, 0.0, 0.0])
                    
                    # Pad point coordinates to match max_points
                    num_sample_points = len(flat_points) // 3
                    while num_sample_points < max_points:
                        row.extend([0.0, 0.0, 0.0])
                        num_sample_points += 1
                
                if max_hit_order_len > 0:
                    order = sample.get('hit_order', [])
                    row.append(len(order))
                    for value in order:
                        row.append(value)
                    while len(order) < max_hit_order_len:
                        row.append(-1)
                
                # Pad to match header length if necessary (shouldn't be needed, but safety check)
                while len(row) < len(header):
                    row.append(0.0)
                
                writer.writerow(row)
        
        print(f"Saved {len(self.samples)} samples to {path}")
    
    def save_as_json(self, path: str, append: bool = False):
        """
        Save the dataset to a JSON file.
        
        The JSON format will be:
        {
            "metadata": {
                "total_samples": 10
            },
            "samples": [
                {
                    "id": 1,
                    "label": "hello",
                    "points_left": [{"index": 0, "x": 0.5, "y": 0.4, "z": 0.2}, ...],
                    "points_right": [{"index": 0, "x": 0.5, "y": 0.4, "z": 0.2}, ...],
                    "hit_order": [0, 5, 12, ...],
                    "chain_code": [...]
                },
                ...
            ]
        }
        
        Args:
            path: File path to save to
            append: If True, append only the latest sample to existing file (default: False)
        """
        if not self.samples:
            print("No samples to save.")
            return
        
        # If appending, only append new samples (avoid duplicates)
        existing_samples = []
        if append and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_samples = existing_data.get('samples', [])
                    
                    # Get existing sample IDs to avoid duplicates
                    existing_ids = {s.get('id') for s in existing_samples if isinstance(s, dict) and 'id' in s}
                    
                    # Only add samples that don't already exist in the file
                    new_samples = [s for s in self.samples if s.get('id') not in existing_ids]
                    
                    if new_samples:
                        all_samples = existing_samples + new_samples
                    else:
                        # No new samples, just update metadata
                        all_samples = existing_samples
            except (json.JSONDecodeError, FileNotFoundError, KeyError):
                # If file format is incompatible, start fresh
                all_samples = self.samples
        else:
            all_samples = self.samples
        
        # Create output structure
        output = {
            'metadata': {
                'total_samples': len(all_samples)
            },
            'samples': all_samples
        }
        
        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        num_new = len(all_samples) - len(existing_samples)
        print(f"Saved {num_new} new sample(s) to {path} (total: {len(all_samples)})")
    
    def load_from_json(self, path: str):
        """
        Load samples from a JSON file.
        
        Args:
            path: File path to load from
        """
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                samples = data.get('samples', [])
                
                # Update next_id based on loaded samples
                if samples:
                    max_id = max(s.get('id', 0) for s in samples)
                    self.next_id = max_id + 1
                
                self.samples = samples
                print(f"Loaded {len(samples)} samples from {path}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading JSON file: {e}")

