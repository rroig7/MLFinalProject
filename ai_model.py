import random
import numpy as np
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural network model
        model = tf.keras.Sequential([
            # First convolutional layer with padding="same" to preserve spatial dimensions
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.state_size),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),  # Use padding='same' to prevent shrinking too fast

            # Second convolutional layer
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),  # Keep pooling layers safe from shrinking too much

            # Flatten the result before passing it to dense layers
            tf.keras.layers.Flatten(),

            # Fully connected layer
            tf.keras.layers.Dense(128, activation='relu'),

            # Output layer for Q-values
            tf.keras.layers.Dense(self.action_size, activation='linear')  # Q-values for actions
        ])

        # Print the summary of the model to see the shape at each layer
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (explore)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Best action (exploit)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Reshape next_state to (1, 60, 80, 1) for prediction
                next_state = np.reshape(next_state, (1, 60, 80, 1))  # Ensure next_state has the correct shape
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # Reshape state to (1, 60, 80, 1) for prediction
            state = np.reshape(state, (1, 60, 80, 1))  # Ensure state has the correct shape
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the model
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, name):
        self.model.save(name)
