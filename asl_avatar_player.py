"""
ASL Avatar Playback System

Plays back gesture sequences with correct timing and sequencing.
"""

import time
from typing import List, Optional, Callable
from asl_avatar_loader import GestureSequence, FrameData
from asl_avatar_renderer import AvatarRenderer
from asl_avatar_text_parser import TextParser


class GesturePlayer:
    """Plays back gesture sequences with timing."""
    
    def __init__(self, renderer: AvatarRenderer, text_parser: TextParser, playback_speed: float = 1.0):
        """
        Initialize gesture player.
        
        Args:
            renderer: AvatarRenderer instance
            text_parser: TextParser instance
            playback_speed: Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
        """
        self.renderer = renderer
        self.text_parser = text_parser
        self.current_sequence: Optional[GestureSequence] = None
        self.current_frame_index = 0
        self.is_playing = False
        self.playback_start_time = 0.0
        self.sequence_start_time = 0.0
        self.sequences: List[GestureSequence] = []
        self.sequence_index = 0
        self.on_frame_callback: Optional[Callable] = None
        self.playback_speed = playback_speed  # Speed multiplier
    
    def load_text(self, text: str) -> bool:
        """
        Load gestures from text input.
        
        Args:
            text: Input text (e.g., "hello sorry")
        
        Returns:
            True if at least one gesture was loaded
        """
        parsed = self.text_parser.parse_text(text)
        
        sequences = []
        for word, gesture in parsed:
            if gesture is not None:
                sequences.append(gesture)
            else:
                print(f"Warning: No gesture found for '{word}'")
        
        if sequences:
            self.sequences = sequences
            self.sequence_index = 0
            return True
        else:
            print(f"Error: No gestures found for text: '{text}'")
            return False
    
    def start_playback(self):
        """Start playing loaded gestures."""
        if not self.sequences:
            return False
        
        self.is_playing = True
        self.sequence_index = 0
        self.current_sequence = self.sequences[0]
        self.current_frame_index = 0
        self.playback_start_time = time.time()
        self.sequence_start_time = self.playback_start_time
        
        return True
    
    def stop_playback(self):
        """Stop playback."""
        self.is_playing = False
        self.current_sequence = None
        self.current_frame_index = 0
    
    def update(self) -> bool:
        """
        Update playback state and render current frame.
        
        Returns:
            True if still playing, False if finished
        """
        if not self.is_playing or not self.sequences:
            return False
        
        if self.sequence_index >= len(self.sequences):
            # All sequences finished
            self.is_playing = False
            return False
        
        current_time = time.time()
        elapsed = (current_time - self.sequence_start_time) * self.playback_speed
        
        # Get current sequence
        sequence = self.sequences[self.sequence_index]
        
        if not sequence.frames:
            # Empty sequence, move to next
            self.sequence_index += 1
            if self.sequence_index < len(self.sequences):
                self.current_sequence = self.sequences[self.sequence_index]
                self.current_frame_index = 0
                self.sequence_start_time = current_time
            return self.is_playing
        
        # Calculate target frame based on elapsed time
        # Use timestamps if available and valid, otherwise fall back to fps
        if sequence.frames and len(sequence.frames) > 1:
            # Check if timestamps are valid (not all zero, increasing)
            first_timestamp = sequence.frames[0].timestamp
            last_timestamp = sequence.frames[-1].timestamp
            
            # Use timestamps if they're valid and span a reasonable duration
            if last_timestamp > first_timestamp and (last_timestamp - first_timestamp) > 0.1:
                # Find frame based on timestamp (apply speed multiplier)
                target_frame = 0
                for i, frame in enumerate(sequence.frames):
                    # Normalize timestamp to start from 0
                    normalized_timestamp = frame.timestamp - first_timestamp
                    # Apply playback speed
                    if normalized_timestamp <= elapsed:
                        target_frame = i
                    else:
                        break
            else:
                # Timestamps invalid, use fps but with speed adjustment
                # Calculate actual fps from timestamps if possible
                if sequence.fps > 0:
                    # Use fps with playback speed multiplier
                    playback_fps = sequence.fps * self.playback_speed
                    # If fps seems too high (e.g., > 60), cap it
                    if playback_fps > 60:
                        playback_fps = 30 * self.playback_speed
                    target_frame = int(elapsed * playback_fps)
                else:
                    # No fps info, estimate from frame count and duration
                    estimated_fps = len(sequence.frames) / max(sequence.duration, 1.0) if sequence.duration > 0 else 30
                    target_frame = int(elapsed * estimated_fps * self.playback_speed)
        else:
            # No frames or single frame
            target_frame = 0
        
        # Check if we've finished this sequence
        if target_frame >= len(sequence.frames):
            # Move to next sequence
            self.sequence_index += 1
            if self.sequence_index < len(self.sequences):
                self.current_sequence = self.sequences[self.sequence_index]
                self.current_frame_index = 0
                self.sequence_start_time = current_time
                return self.is_playing
            else:
                # All sequences finished
                self.is_playing = False
                return False
        
        # Clamp to valid range
        target_frame = min(target_frame, len(sequence.frames) - 1)
        self.current_frame_index = target_frame
        
        # Get current frame
        frame = sequence.frames[target_frame]
        
        # Set frame number in renderer for debug output
        if hasattr(self.renderer, '_current_frame_num'):
            self.renderer._current_frame_num = target_frame
        
        # Render frame
        self.renderer.render_frame(
            frame.body_pose,
            frame.left_hand,
            frame.right_hand,
            frame.face
        )
        
        # Call callback if set
        if self.on_frame_callback:
            self.on_frame_callback(frame, target_frame, len(sequence.frames))
        
        return True
    
    def get_current_frame_info(self) -> dict:
        """Get information about current playback state."""
        if not self.current_sequence:
            return {
                "sequence": None,
                "frame": 0,
                "total_frames": 0,
                "progress": 0.0
            }
        
        progress = 0.0
        if self.current_sequence.num_frames > 0:
            progress = self.current_frame_index / self.current_sequence.num_frames
        
        return {
            "sequence": self.current_sequence.label,
            "frame": self.current_frame_index,
            "total_frames": self.current_sequence.num_frames,
            "progress": progress,
            "sequence_index": self.sequence_index,
            "total_sequences": len(self.sequences)
        }
