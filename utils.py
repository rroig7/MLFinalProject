import cv2
import numpy as np

def preprocess_screen(screen):
    # Convert to grayscale and resize to speed up processing (if needed)
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    resized_screen = cv2.resize(gray_screen, (80, 60))  # Downsample for simplicity
    return resized_screen

def check_if_crashed(screen):
    # Detect color change or lack of movement in the character's position to detect a crash
    # For simplicity, this assumes a certain region of the screen indicates a crash
    crash_region = screen[50:55, 10:15]  # Example coordinates for a specific crash area
    if np.mean(crash_region) < 50:  # If the pixels are dark, it means a crash happened
        return True
    return False