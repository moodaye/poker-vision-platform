"""
Mouse Controller Module
Handles mouse automation for clicking detected OK buttons.
"""

import time
import logging
from typing import Tuple, List
import config

class MouseController:
    """Class to handle mouse movements and clicks."""
    
    def __init__(self):
        """Initialize the mouse controller."""
        self.last_click_time = 0
        self.click_count = 0
        self.minute_start_time = time.time()
        self._pyautogui = None
        
        logging.info("Mouse controller initialized")
    
    def _get_pyautogui(self):
        """Lazy import pyautogui to avoid X11 issues at module level."""
        if self._pyautogui is None:
            import pyautogui
            self._pyautogui = pyautogui
            # Configure pyautogui settings
            self._pyautogui.FAILSAFE = True
            self._pyautogui.PAUSE = config.CLICK_DELAY
        return self._pyautogui
    
    def click_button(self, x: int, y: int) -> bool:
        """
        Click at the specified coordinates with safety checks.
        
        Args:
            x: X coordinate to click
            y: Y coordinate to click
            
        Returns:
            bool: True if click was successful, False otherwise
        """
        try:
            # Safety check: rate limiting
            if not self._check_rate_limit():
                logging.warning("Rate limit exceeded, skipping click")
                return False
            
            # Safety check: minimum time between clicks
            current_time = time.time()
            if current_time - self.last_click_time < config.CLICK_COOLDOWN:
                logging.info("Click cooldown active, skipping click")
                return False
            
            # Safety check: coordinates are within screen bounds
            pyautogui = self._get_pyautogui()
            screen_width, screen_height = pyautogui.size()
            if not (0 <= x <= screen_width and 0 <= y <= screen_height):
                logging.error(f"Click coordinates ({x}, {y}) are outside screen bounds")
                return False
            
            # Perform the click
            logging.info(f"Clicking at coordinates ({x}, {y})")
            
            # Move mouse to position
            pyautogui.moveTo(x, y, duration=0.1)
            
            # Small pause before clicking
            time.sleep(0.05)
            
            # Perform click
            pyautogui.click(x, y)
            
            # Update tracking variables
            self.last_click_time = current_time
            self.click_count += 1
            
            logging.info(f"Successfully clicked at ({x}, {y})")
            return True
            
        except Exception as e:
            # Handle FailSafeException and other pyautogui exceptions
            if "FailSafe" in str(type(e)):
                logging.info("PyAutoGUI fail-safe triggered, stopping clicks")
                return False
            logging.error(f"Failed to click at ({x}, {y}): {e}")
            return False
    
    def click_multiple_buttons(self, button_locations: List[Tuple[int, int, float]], 
                             max_clicks: int = 1) -> int:
        """
        Click multiple detected buttons with safety limits.
        
        Args:
            button_locations: List of (x, y, confidence) tuples
            max_clicks: Maximum number of buttons to click
            
        Returns:
            int: Number of successful clicks
        """
        successful_clicks = 0
        
        # Limit the number of clicks
        clicks_to_perform = min(len(button_locations), max_clicks)
        
        for i in range(clicks_to_perform):
            x, y, confidence = button_locations[i]
            
            if self.click_button(x, y):
                successful_clicks += 1
                
                # Add delay between multiple clicks
                if i < clicks_to_perform - 1:
                    time.sleep(config.CLICK_COOLDOWN)
            else:
                # If one click fails, stop trying others for safety
                break
        
        return successful_clicks
    
    def _check_rate_limit(self) -> bool:
        """
        Check if we're within the rate limit for clicks.
        
        Returns:
            bool: True if within rate limit, False otherwise
        """
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - self.minute_start_time >= 60:
            self.click_count = 0
            self.minute_start_time = current_time
        
        return self.click_count < config.MAX_CLICKS_PER_MINUTE
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """
        Get the current mouse position.
        
        Returns:
            Tuple of (x, y) coordinates
        """
        pyautogui = self._get_pyautogui()
        pos = pyautogui.position()
        return (int(pos.x), int(pos.y))
    
    def move_mouse(self, x: int, y: int, duration: float = 0.1):
        """
        Move mouse to specified position without clicking.
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration: Time to take for the movement
        """
        try:
            pyautogui = self._get_pyautogui()
            pyautogui.moveTo(x, y, duration=duration)
        except Exception as e:
            logging.error(f"Failed to move mouse to ({x}, {y}): {e}")
    
    def reset_rate_limit(self):
        """Reset the rate limiting counters."""
        self.click_count = 0
        self.minute_start_time = time.time()
        logging.info("Rate limit counters reset")
