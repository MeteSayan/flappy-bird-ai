import sys
import os
from pathlib import Path
import numpy as np
import torch
import pygame

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from ai_agent.model import DQNAgent
from game.flappy_bird import Game

def preprocess_state(state: dict) -> np.ndarray:
    """Convert game state dictionary to numpy array"""
    return np.array([
        state['bird_y'] / 600.0,  # Normalize y position
        state['bird_velocity'] / 10.0,  # Normalize velocity
        state['next_pipe_x'] / 400.0,  # Normalize pipe x position
        state['next_pipe_gap_y'] / 600.0  # Normalize pipe gap y position
    ])

def play_with_ai():
    # Initialize game and agent
    game = Game()
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    
    # Load the best model
    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        print("No trained model found! Please train the AI first.")
        return
    
    print(f"Loading model from {model_path}")
    agent.load(model_path)
    agent.epsilon = 0  # Set epsilon to 0 for pure exploitation (no random actions)
    
    print("\nAI Demonstration Controls:")
    print("Q - Quit")
    print("R - Restart")
    print("Space - Manual flap (if you want to play yourself)")
    print("\nWatching AI play...")
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset_game()
                elif event.key == pygame.K_SPACE:
                    game.bird.flap()
        
        # Get current state and AI's action
        state = preprocess_state(game.get_state())
        action = agent.act(state)
        
        # Apply action
        if action == 1:  # Flap
            game.bird.flap()
        
        # Update game state
        game.update()
        
        # Draw everything
        game.draw()
        game.clock.tick(60)
        
        # Display current score
        if game.game_over:
            print(f"Game Over! Score: {game.score}")
            print("Press R to restart or Q to quit")

if __name__ == "__main__":
    try:
        play_with_ai()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pygame.quit() 