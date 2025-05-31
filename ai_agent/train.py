import sys
import os
from pathlib import Path
import numpy as np
import torch
import threading
import queue
import time

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from ai_agent.model import DQNAgent
from game.flappy_bird import Game

class TrainingThread(threading.Thread):
    """
    Thread for running the AI training process.
    Handles the training loop and model saving while keeping the game responsive.
    """
    def __init__(self, game, episodes=2000, save_interval=100):
        super().__init__()
        self.game = game
        self.episodes = episodes
        self.save_interval = save_interval
        self.running = True
        self.paused = False
        self.state_size = 4
        self.action_size = 2
        
        # Initialize the DQN agent with optimized parameters
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_size=128,  # Increased for better learning capacity
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=100000,  # Large memory for better experience replay
            batch_size=128  # Larger batch size for stable learning
        )
        self.progress_queue = queue.Queue()
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        self.best_score = 0
        self.scores = []
        self.daemon = True

    def preprocess_state(self, state: dict) -> np.ndarray:
        """
        Convert game state dictionary to normalized numpy array.
        Normalizes values to [0, 1] range for better neural network performance.
        """
        return np.array([
            state['bird_y'] / 600.0,
            state['bird_velocity'] / 10.0,
            state['next_pipe_x'] / 400.0,
            state['next_pipe_gap_y'] / 600.0
        ])

    def stop(self):
        """Stop the training thread"""
        self.running = False

    def pause(self):
        """Pause the training process"""
        self.paused = True

    def resume(self):
        """Resume the training process"""
        self.paused = False

    def run(self):
        """
        Main training loop.
        Runs episodes and updates the AI agent based on game experience.
        """
        for episode in range(self.episodes):
            if not self.running:
                break

            while self.paused:
                time.sleep(0.1)
                if not self.running:
                    return

            self.game.reset_game()
            state = self.preprocess_state(self.game.get_state())
            total_reward = 0
            steps = 0
            consecutive_pipes = 0
            
            while not self.game.game_over and self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue

                # Agent selects action
                action = self.agent.act(state)
                
                # Apply action
                if action == 1:  # Flap
                    self.game.bird.flap()
                
                # Update game state
                self.game.update()
                next_state = self.preprocess_state(self.game.get_state())
                
                # Calculate reward with improved structure
                reward = 0
                if self.game.score > steps:  # Passed a pipe
                    reward = 2.0
                    consecutive_pipes += 1
                    # Bonus for consecutive pipes
                    if consecutive_pipes > 1:
                        reward += 1.0 * (consecutive_pipes - 1)
                elif self.game.game_over:  # Died
                    reward = -2.0
                    consecutive_pipes = 0
                else:
                    # Reward for staying alive and moving towards the gap
                    bird_y = self.game.bird.y
                    gap_y = self.game.pipes[0].gap_y if self.game.pipes else 300
                    if abs(bird_y - gap_y) < 100:  # Close to the gap
                        reward = 0.2
                
                # Store experience in replay memory
                self.agent.memory.append((state, action, reward, next_state, self.game.game_over))
                
                # Train the agent
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.train()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Update target network
                if steps % 10 == 0:
                    self.agent.update_target_network()
            
            # Update best score and save model if improved
            if self.game.score > self.best_score:
                self.best_score = self.game.score
                self.agent.save(f'models/best_model.pth')
                print(f"\nNew best score: {self.best_score}! Model saved.")
            
            self.scores.append(self.game.score)
            avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
            
            # Put progress information in the queue
            self.progress_queue.put({
                'episode': episode + 1,
                'score': self.game.score,
                'best_score': self.best_score,
                'avg_score': avg_score,
                'epsilon': self.agent.epsilon,
                'total_reward': total_reward
            })
            
            # Save model periodically
            if (episode + 1) % self.save_interval == 0:
                self.agent.save(f'models/model_episode_{episode + 1}.pth')

def train(episodes: int = 2000, save_interval: int = 100):
    """
    Initialize and start the training process.
    
    Args:
        episodes: Number of episodes to train
        save_interval: How often to save model checkpoints
    """
    game = Game()
    training_thread = TrainingThread(game, episodes, save_interval)
    training_thread.start()
    
    # Main game loop - minimized visualization
    while True:
        game.handle_events()
        game.update()
        # Minimize drawing operations
        if training_thread.best_score > 0:  # Only draw if we have a score
            game.draw()
        game.clock.tick(120)  # Increased frame rate
        
        # Check for training progress
        try:
            while True:  # Process all available progress updates
                progress = training_thread.progress_queue.get_nowait()
                print(f"\rEpisode: {progress['episode']}/{episodes} | "
                      f"Score: {progress['score']} | "
                      f"Best: {progress['best_score']} | "
                      f"Avg: {progress['avg_score']:.2f} | "
                      f"Epsilon: {progress['epsilon']:.2f}", end="")
        except queue.Empty:
            pass

if __name__ == "__main__":
    train() 