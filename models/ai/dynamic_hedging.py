import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
import pandas as pd
from collections import deque
import random

class DynamicHedgingAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000):
        """
        Dynamic hedging agent using Deep Q-Learning
        
        Args:
            state_dim: Dimension of state space (market features)
            action_dim: Dimension of action space (hedging positions)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for exploration
            epsilon_min: Minimum exploration rate
            memory_size: Size of replay memory
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.set_weights(self.q_network.get_weights())
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    def _build_network(self) -> tf.keras.Model:
        """Builds the Q-network for hedging decisions"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_dim)
        ])
        return model
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Stores experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Selects action based on current state using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    @tf.function
    def _train_step(self, states: tf.Tensor, actions: tf.Tensor, 
                    rewards: tf.Tensor, next_states: tf.Tensor, dones: tf.Tensor) -> tf.Tensor:
        """Single training step using temporal difference learning"""
        with tf.GradientTape() as tape:
            # Current Q-values
            current_q = self.q_network(states)
            current_q_actions = tf.reduce_sum(
                current_q * tf.one_hot(actions, self.action_dim),
                axis=1
            )
            
            # Target Q-values
            next_q = self.target_network(next_states)
            max_next_q = tf.reduce_max(next_q, axis=1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
            # Loss calculation
            loss = tf.reduce_mean(tf.square(target_q - current_q_actions))
        
        # Gradient update
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss
    
    def train(self, batch_size: int = 32) -> Optional[float]:
        """Trains the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return None
            
        # Sample batch from memory
        batch = random.sample(self.memory, batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Train step
        loss = self._train_step(states, actions, rewards, next_states, dones)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.numpy()
    
    def update_target_network(self):
        """Updates target network weights with current Q-network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def calculate_hedge_ratio(self, state: np.ndarray) -> float:
        """Calculates the optimal hedge ratio for the current state"""
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        action = np.argmax(q_values[0])
        
        # Convert discrete action to continuous hedge ratio
        hedge_ratio = action / (self.action_dim - 1)  # Scale to [0, 1]
        return hedge_ratio
    
    def save_model(self, path: str):
        """Saves the Q-network and target network weights"""
        self.q_network.save_weights(path + '_q_network')
        self.target_network.save_weights(path + '_target_network')
        
    def load_model(self, path: str):
        """Loads the Q-network and target network weights"""
        self.q_network.load_weights(path + '_q_network')
        self.target_network.load_weights(path + '_target_network') 