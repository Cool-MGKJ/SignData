"""
Text Parser for ASL Avatar Playback

Maps text input to gesture sequences.
"""

import re
from typing import List, Tuple, Optional
from asl_avatar_loader import GestureLoader, GestureSequence


class TextParser:
    """Parses text input and maps to gesture sequences."""
    
    def __init__(self, gesture_loader: GestureLoader):
        """
        Initialize text parser.
        
        Args:
            gesture_loader: GestureLoader instance for loading gestures
        """
        self.gesture_loader = gesture_loader
        self._available_gestures = set(gesture_loader.list_available_gestures())
    
    def parse_text(self, text: str) -> List[Tuple[str, Optional[GestureSequence]]]:
        """
        Parse text input and map to gesture sequences.
        
        Args:
            text: Input text (e.g., "hello sorry", "thank you")
        
        Returns:
            List of (word, gesture_sequence) tuples
            If a word doesn't map to a gesture, sequence will be None
        """
        # Normalize text
        text = text.lower().strip()
        
        # Split into words (handle multi-word gestures)
        # First, try to match known multi-word gestures
        words = []
        remaining_text = text
        
        # Sort by length (longest first) to match multi-word gestures first
        available_sorted = sorted(self._available_gestures, key=len, reverse=True)
        
        while remaining_text:
            matched = False
            for gesture_label in available_sorted:
                # Try matching gesture label (with underscores or spaces)
                gesture_normalized = gesture_label.replace("_", " ")
                if remaining_text.startswith(gesture_normalized):
                    # Check if it's a complete word/phrase
                    next_char_idx = len(gesture_normalized)
                    if (next_char_idx >= len(remaining_text) or 
                        remaining_text[next_char_idx] in [' ', ',', '.', '!', '?']):
                        words.append(gesture_label)
                        remaining_text = remaining_text[next_char_idx:].strip()
                        matched = True
                        break
            
            if not matched:
                # Extract single word
                match = re.match(r'(\w+)', remaining_text)
                if match:
                    word = match.group(1)
                    words.append(word)
                    remaining_text = remaining_text[len(word):].strip()
                else:
                    # Skip non-word characters
                    remaining_text = remaining_text[1:].strip()
        
        # Map words to gestures
        result = []
        for word in words:
            # Try exact match first
            gesture = self.gesture_loader.load_gesture(word)
            
            # If not found, try with underscores
            if gesture is None:
                word_underscore = word.replace(" ", "_")
                gesture = self.gesture_loader.load_gesture(word_underscore)
            
            result.append((word, gesture))
        
        return result
    
    def update_available_gestures(self):
        """Update the list of available gestures."""
        self._available_gestures = set(self.gesture_loader.list_available_gestures())
