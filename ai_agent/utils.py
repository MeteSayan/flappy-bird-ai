import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

def plot_training_progress(scores: List[int], avg_scores: List[float], 
                          epsilon_values: List[float], save_path: str = None):
    """
    Plot training progress including scores, average scores, and epsilon values.
    
    Args:
        scores: List of scores for each episode
        avg_scores: List of average scores (typically over last 100 episodes)
        epsilon_values: List of epsilon values over time
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot scores
    plt.subplot(1, 2, 1)
    plt.plot(scores, label='Score', alpha=0.6)
    plt.plot(avg_scores, label='Average Score', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    
    # Plot epsilon
    plt.subplot(1, 2, 2)
    plt.plot(epsilon_values, label='Epsilon', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_reward(score: int, game_over: bool, steps: int) -> float:
    """
    Calculate reward for the agent based on game state.
    
    Args:
        score: Current game score
        game_over: Whether the game is over
        steps: Number of steps taken
        
    Returns:
        float: Calculated reward
    """
    if game_over:
        return -1.0  # Penalty for dying
    elif score > steps:
        return 1.0  # Reward for passing a pipe
    return 0.1  # Small reward for staying alive

def normalize_state(state: dict) -> np.ndarray:
    """
    Normalize the game state values to be between 0 and 1.
    
    Args:
        state: Dictionary containing game state
        
    Returns:
        np.ndarray: Normalized state array
    """
    return np.array([
        state['bird_y'] / 600.0,  # Normalize y position
        state['bird_velocity'] / 10.0,  # Normalize velocity
        state['next_pipe_x'] / 400.0,  # Normalize pipe x position
        state['next_pipe_gap_y'] / 600.0  # Normalize pipe gap y position
    ])

def save_training_metrics(scores: List[int], avg_scores: List[float], 
                         epsilon_values: List[float], path: str):
    """
    Save training metrics to a file.
    
    Args:
        scores: List of scores
        avg_scores: List of average scores
        epsilon_values: List of epsilon values
        path: Path to save the metrics
    """
    metrics = {
        'scores': scores,
        'avg_scores': avg_scores,
        'epsilon_values': epsilon_values
    }
    np.save(path, metrics)

def load_training_metrics(path: str) -> Tuple[List[int], List[float], List[float]]:
    """
    Load training metrics from a file.
    
    Args:
        path: Path to the metrics file
        
    Returns:
        Tuple containing scores, average scores, and epsilon values
    """
    metrics = np.load(path, allow_pickle=True).item()
    return metrics['scores'], metrics['avg_scores'], metrics['epsilon_values'] 