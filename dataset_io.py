"""
Dataset input/output module.

This module handles storing and saving labelled ASL sign samples
in both CSV and JSON formats.
"""

import json
import csv
import os
from typing import List, Tuple, Optional
from datetime import datetime


class DatasetManager:
    """Manages in-memory dataset and file I/O for ASL samples."""
    
    def __init__(self):
        """Initialize an empty dataset."""
        self.samples = []  # List of sample dicts
        self.next_id = 1
    
    def add_sample(
        self,
        label: str,
        normalized_points: List[Tuple[float, float, float]],
        hand: str = "unknown",
        hit_grid_vector: Optional[List[int]] = None,
        hit_points_coordinates: Optional[List[Tuple[float, float, float]]] = None
    ) -> int:
        """
        Add a new sample to the dataset.
        
        Args:
            label: The ASL sign label (e.g., "hello", "thank you")
            normalized_points: List of normalized (x, y, z) tuples
            hand: Which hand(s) detected ("left", "right", "both", "unknown")
            hit_grid_vector: Optional list of 0s and 1s representing which grid points were hit
            hit_points_coordinates: Optional list of (x, y, z) tuples for hit grid points
        
        Returns:
            The unique sample ID assigned to this sample
        """
        sample_id = self.next_id
        self.next_id += 1
        
        # Flatten the 3D points into a single list: [x1, y1, z1, x2, y2, z2, ...]
        flattened_points = []
        for point in normalized_points:
            flattened_points.extend([point[0], point[1], point[2]])
        
        sample = {
            'id': sample_id,
            'label': label,
            'hand': hand,
            'points': flattened_points,
            'num_points': len(normalized_points),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add hit grid vector if provided (for backward compatibility)
        if hit_grid_vector is not None:
            sample['hit_grid'] = hit_grid_vector
            sample['num_grid_points'] = len(hit_grid_vector)
            sample['num_hit_points'] = sum(hit_grid_vector)
        
        # Add hit points with coordinates (primary data)
        if hit_points_coordinates is not None:
            # Flatten hit points coordinates: [x1, y1, z1, x2, y2, z2, ...]
            flattened_hit_points = []
            for point in hit_points_coordinates:
                flattened_hit_points.extend([point[0], point[1], point[2]])
            sample['hit_points'] = flattened_hit_points
            sample['hit_points_count'] = len(hit_points_coordinates)
        
        self.samples.append(sample)
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
        
        # Get maximum grid size (if any samples have hit_grid)
        max_grid_points = 0
        has_grid_data = False
        for s in self.samples:
            if 'hit_grid' in s:
                has_grid_data = True
                grid_size = len(s['hit_grid'])
                if grid_size > max_grid_points:
                    max_grid_points = grid_size
        
        # Create header
        header = ['id', 'label', 'hand', 'num_points']
        for i in range(max_points):
            header.extend([f'point_{i}_x', f'point_{i}_y', f'point_{i}_z'])
        
        # Add grid columns if any samples have grid data
        if has_grid_data:
            header.append('num_grid_points')
            header.append('num_hit_points')
            for i in range(max_grid_points):
                header.append(f'g_{i}')
        
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
                
                # Add all point coordinates
                points = sample['points']
                for i in range(0, len(points), 3):
                    if i + 2 < len(points):
                        row.extend([points[i], points[i+1], points[i+2]])
                    else:
                        # Pad if needed
                        row.extend([0.0, 0.0, 0.0])
                
                # Pad point coordinates to match max_points
                num_sample_points = len(points) // 3
                while num_sample_points < max_points:
                    row.extend([0.0, 0.0, 0.0])
                    num_sample_points += 1
                
                # Add grid data if present
                if has_grid_data:
                    if 'hit_grid' in sample:
                        row.append(sample.get('num_grid_points', len(sample['hit_grid'])))
                        row.append(sample.get('num_hit_points', sum(sample['hit_grid'])))
                        # Add grid vector
                        grid_vec = sample['hit_grid']
                        row.extend(grid_vec)
                        # Pad to max_grid_points if needed
                        while len(grid_vec) < max_grid_points:
                            row.append(0)
                    else:
                        # No grid data for this sample
                        row.append(0)  # num_grid_points
                        row.append(0)  # num_hit_points
                        # Pad grid columns
                        for _ in range(max_grid_points):
                            row.append(0)
                
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
            "samples": [
                {
                    "id": 1,
                    "label": "hello",
                    "hand": "right",
                    "num_points": 21,
                    "points": [x1, y1, z1, x2, y2, z2, ...],
                    "timestamp": "..."
                },
                ...
            ]
        }
        
        Args:
            path: File path to save to
            append: If True, append to existing file (default: False)
        """
        if not self.samples:
            print("No samples to save.")
            return
        
        # If appending and file exists, load existing data
        if append and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_samples = existing_data.get('samples', [])
                    # Merge with new samples
                    all_samples = existing_samples + self.samples
            except (json.JSONDecodeError, FileNotFoundError):
                all_samples = self.samples
        else:
            all_samples = self.samples
        
        # Create output structure
        output = {
            'metadata': {
                'total_samples': len(all_samples),
                'exported_at': datetime.now().isoformat(),
                'format_version': '1.0'
            },
            'samples': all_samples
        }
        
        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.samples)} samples to {path} (total: {len(all_samples)})")
    
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

