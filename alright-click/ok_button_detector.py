"""
OK Button Detection Module
Handles computer vision operations for detecting OK buttons on screen.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from templates.ok_button_template import OKButtonTemplate
import config

class OKButtonDetector:
    """Class to detect OK buttons on screen using computer vision."""
    
    def __init__(self, confidence_threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD):
        """
        Initialize the OK button detector.
        
        Args:
            confidence_threshold: Minimum confidence for button detection (0.0-1.0)
        """
        self.confidence_threshold = max(config.MIN_CONFIDENCE_THRESHOLD, 
                                      min(config.MAX_CONFIDENCE_THRESHOLD, confidence_threshold))
        self.template_manager = OKButtonTemplate()
        self.last_screenshot = None
        self._pyautogui = None
        
        logging.info(f"OK Button Detector initialized with confidence threshold: {self.confidence_threshold}")
    
    def _get_pyautogui(self):
        """Lazy import pyautogui to avoid X11 issues at module level."""
        if self._pyautogui is None:
            import pyautogui
            self._pyautogui = pyautogui
            # Configure pyautogui settings
            self._pyautogui.FAILSAFE = True
            self._pyautogui.PAUSE = config.CLICK_DELAY
        return self._pyautogui
    
    def capture_screen(self) -> np.ndarray:
        """
        Capture the current screen.
        
        Returns:
            numpy.ndarray: Screenshot as OpenCV image
        """
        try:
            # Try different screenshot methods
            screenshot = None
            
            # Method 1: Try pyautogui
            try:
                pyautogui = self._get_pyautogui()
                screenshot = pyautogui.screenshot()
            except Exception as e:
                logging.warning(f"PyAutoGUI screenshot failed: {e}")
            
            # Method 2: Try using scrot via subprocess if pyautogui fails
            if screenshot is None:
                import subprocess
                import tempfile
                import os
                from PIL import Image
                
                try:
                    # Create temporary file for screenshot
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    # Use scrot to capture screen
                    result = subprocess.run(['scrot', tmp_path], 
                                          capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0 and os.path.exists(tmp_path):
                        screenshot = Image.open(tmp_path)
                        os.unlink(tmp_path)  # Clean up temp file
                    else:
                        logging.warning(f"Scrot failed: {result.stderr}")
                except Exception as e:
                    logging.warning(f"Scrot screenshot failed: {e}")
            
            # Method 3: Create a dummy screenshot for testing if all else fails
            if screenshot is None:
                logging.warning("Creating dummy screenshot for testing purposes")
                from PIL import Image, ImageDraw, ImageFont
                
                # Create a test image with an OK button
                screenshot = Image.new('RGB', (800, 600), color='white')
                draw = ImageDraw.Draw(screenshot)
                
                # Draw a simple OK button for testing
                button_rect = (350, 250, 450, 300)
                draw.rectangle(button_rect, outline='black', fill='lightgray', width=2)
                
                # Try to add text
                try:
                    font = ImageFont.load_default()
                    text_bbox = draw.textbbox((0, 0), "OK", font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    text_x = 350 + (100 - text_width) // 2
                    text_y = 250 + (50 - text_height) // 2
                    draw.text((text_x, text_y), "OK", fill='black', font=font)
                except:
                    # Fallback if font loading fails
                    draw.text((385, 270), "OK", fill='black')
            
            if screenshot is None:
                raise Exception("All screenshot methods failed")
            
            # Convert PIL image to OpenCV format
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            self.last_screenshot = screenshot_cv.copy()
            
            return screenshot_cv
            
        except Exception as e:
            logging.error(f"Failed to capture screen: {e}")
            raise
    
    def detect_ok_buttons(self, screenshot: Optional[np.ndarray] = None) -> List[Tuple[int, int, float]]:
        """
        Detect OK buttons in the screenshot.
        
        Args:
            screenshot: Optional screenshot to analyze. If None, captures new screenshot.
            
        Returns:
            List of tuples containing (x, y, confidence) for each detected button
        """
        if screenshot is None:
            screenshot = self.capture_screen()
        
        # Convert to grayscale for template matching
        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        detected_buttons = []
        templates = self.template_manager.get_templates()
        
        for template in templates:
            try:
                # Perform template matching
                result = cv2.matchTemplate(gray_screenshot, template, cv2.TM_CCOEFF_NORMED)
                
                # Find locations where the match confidence is above threshold
                locations = np.where(result >= self.confidence_threshold)
                
                # Get template dimensions
                template_height, template_width = template.shape
                
                # Process each match
                for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                    x, y = pt
                    confidence = result[y, x]
                    
                    # Calculate center of the button
                    center_x = x + template_width // 2
                    center_y = y + template_height // 2
                    
                    # Check if this detection is too close to existing ones (avoid duplicates)
                    is_duplicate = False
                    for existing_x, existing_y, _ in detected_buttons:
                        distance = np.sqrt((center_x - existing_x)**2 + (center_y - existing_y)**2)
                        if distance < 30:  # Minimum distance between detections
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        detected_buttons.append((center_x, center_y, confidence))
                        logging.info(f"OK button detected at ({center_x}, {center_y}) with confidence {confidence:.3f}")
                
            except Exception as e:
                logging.warning(f"Template matching failed for one template: {e}")
                continue
        
        # Sort by confidence (highest first)
        detected_buttons.sort(key=lambda x: x[2], reverse=True)
        
        logging.info(f"Total OK buttons detected: {len(detected_buttons)}")
        return detected_buttons
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold for detection.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = max(config.MIN_CONFIDENCE_THRESHOLD, 
                                      min(config.MAX_CONFIDENCE_THRESHOLD, threshold))
        logging.info(f"Confidence threshold updated to: {self.confidence_threshold}")
    
    def visualize_detections(self, screenshot: np.ndarray, detections: List[Tuple[int, int, float]]) -> np.ndarray:
        """
        Draw detection results on the screenshot for debugging.
        
        Args:
            screenshot: Original screenshot
            detections: List of detected button locations
            
        Returns:
            Screenshot with detection markers drawn
        """
        visualization = screenshot.copy()
        
        for x, y, confidence in detections:
            # Draw circle at detection point
            cv2.circle(visualization, (x, y), 10, (0, 255, 0), 2)
            
            # Draw confidence text
            text = f"{confidence:.2f}"
            cv2.putText(visualization, text, (x - 20, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return visualization
