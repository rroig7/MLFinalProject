import cv2
from mss import mss
import numpy as np

# Define the screen region to capture (adjust according to your game window)
bbox = {"top": 100, "left": 100, "width": 800, "height": 600}

def capture_screen():
    with mss() as sct:
        screen = np.array(sct.grab(bbox))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        return screen