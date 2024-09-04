import numpy as np
import cv2
from mss import mss
from ai_model import DQNAgent  # Your existing AI model
import time

# Define the region of the screen where the player square is located
bbox = {"top": 100, "left": 100, "width": 80, "height": 60}


# Capture game frame using mss
def capture_game_frame():
    """Captures the game screen and converts it to grayscale."""
    with mss() as sct:
        frame = np.array(sct.grab(bbox))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"Captured frame shape: {gray_frame.shape}")  # Print the shape of the captured frame
        return gray_frame




# Crash detection based on pixel changes
def detect_crash(prev_frame, current_frame):
    """Detects if a crash has occurred based on pixel changes."""
    # The frames are already in grayscale, so no need for cvtColor
    prev_gray = prev_frame
    current_gray = current_frame

    # Calculate absolute difference between consecutive frames
    diff = cv2.absdiff(prev_gray, current_gray)

    # Threshold the difference to detect significant pixel changes
    _, diff_thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Count the number of non-zero pixels (signifying change)
    non_zero_count = np.count_nonzero(diff_thresh)

    # If significant change detected, consider it a crash
    if non_zero_count > 500:  # Adjust threshold for the game
        return True
    return False


# Initialize your AI agent
state_size = (60, 80, 1)  # Input size based on game frame dimensions
action_size = 2  # Number of possible actions (e.g., jump or no jump)
agent = DQNAgent(state_size, action_size)


# Game loop
def run_game_ai():
    prev_frame = capture_game_frame()  # Initial grayscale frame
    episode = 0

    for episode in range(1000):  # Number of episodes/games to play
        print(f"Episode {episode + 1}/1000")
        done = False
        total_reward = 0

        # Reshape the captured frame to (1, 60, 80, 1)
        state = np.reshape(prev_frame, (1, 60, 80, 1))  # Initial state

        while not done:
            # Capture the current game frame
            current_frame = capture_game_frame()

            # Detect crash
            crash_detected = detect_crash(prev_frame, current_frame)
            if crash_detected:
                print("Crash detected!")
                reward = -100  # Negative reward for crashing
                done = True
            else:
                reward = 1  # Positive reward for surviving

            # Store the experience in the agent's memory
            next_state = np.reshape(current_frame, (1, 60, 80, 1))  # Reshape for AI input

            # Use your AI to decide an action (0 = no jump, 1 = jump)
            action = agent.act(state)

            # Perform the action (jump or no jump)
            # perform_action(action)  # You would define this function

            # Store the experience and train the agent
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > 32:  # Assuming batch size is 32
                agent.replay(32)

            # Update total reward and move to next state
            total_reward += reward
            state = next_state
            prev_frame = current_frame  # Update for crash detection

            if done:
                print(f"Episode finished with total reward: {total_reward}")
                break

            # Add a small delay to match game speed (adjust as necessary)
            time.sleep(0.05)


# Run the AI
if __name__ == "__main__":
    run_game_ai()
