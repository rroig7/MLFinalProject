import pyautogui

# Map actions to actual keypresses
def perform_action(action):
    if action == 1:  # 1 means jump
        pyautogui.press('space')