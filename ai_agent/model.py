import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model architecture.
    A neural network that approximates the Q-value function for the Flappy Bird game.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Learning agent for playing Flappy Bird.
    Implements experience replay and target network for stable learning.
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 100000,
                 batch_size: int = 64):
        """
        Initialize the DQN agent with specified parameters.
        
        Args:
            state_size: Number of state features
            action_size: Number of possible actions
            hidden_size: Number of neurons in hidden layers
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            memory_size: Size of the replay memory
            batch_size: Number of experiences to sample for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy and target networks
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state: np.ndarray) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            
        Returns:
            Selected action (0: do nothing, 1: flap)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def train(self) -> None:
        """
        Train the policy network using experience replay.
        Samples random experiences from memory and updates the network.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample random experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays first, then to tensors for better performance
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update network
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self) -> None:
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        """Save the policy network to a file"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the policy network from a file"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 