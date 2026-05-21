"""
Template generation for OK button detection.
This module creates template images for matching OK buttons on screen.
"""

import cv2
import numpy as np
from typing import List, Tuple

class OKButtonTemplate:
    """Class to generate and manage OK button templates."""
    
    def __init__(self):
        """Initialize the template generator."""
        self.templates = []
        self._generate_templates()
    
    def _generate_templates(self):
        """Generate various OK button templates programmatically."""
        # Generate different sizes and styles of OK buttons
        self.templates = [
            self._create_text_template("OK", (60, 25)),
            self._create_text_template("OK", (80, 30)),
            self._create_text_template("OK", (100, 35)),
            self._create_text_template("Ok", (60, 25)),
            self._create_text_template("Ok", (80, 30)),
            self._create_button_template("OK", (80, 30), with_border=True),
            self._create_button_template("OK", (100, 35), with_border=True),
            self._create_button_template("Ok", (80, 30), with_border=True),
        ]
    
    def _create_text_template(self, text: str, size: Tuple[int, int]) -> np.ndarray:
        """Create a simple text template."""
        width, height = size
        template = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color = (0, 0, 0)  # Black text
        
        # Get text size and center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(template, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    def _create_button_template(self, text: str, size: Tuple[int, int], with_border: bool = True) -> np.ndarray:
        """Create a button-style template with border."""
        width, height = size
        template = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        if with_border:
            # Add button border
            cv2.rectangle(template, (0, 0), (width-1, height-1), (100, 100, 100), 1)
            cv2.rectangle(template, (1, 1), (width-2, height-2), (200, 200, 200), 1)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color = (0, 0, 0)  # Black text
        
        # Get text size and center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(template, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    def get_templates(self) -> List[np.ndarray]:
        """Get all available templates."""
        return self.templates
    
    def add_custom_template(self, template: np.ndarray):
        """Add a custom template to the collection."""
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.templates.append(template)
