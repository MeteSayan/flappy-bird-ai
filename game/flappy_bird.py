import pygame
import random
import sys
import threading
from typing import List, Tuple, Optional

# Initialize Pygame
pygame.init()

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.25
FLAP_STRENGTH = -5  # Reduced from -7 for smoother control
PIPE_SPEED = 4      # Increased from 3 for more challenging gameplay
PIPE_GAP = 200      # Increased from 150 for easier passage
PIPE_FREQUENCY = 1000  # Time between pipe spawns (milliseconds)
GROUND_HEIGHT = 100

# Color Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
SKY_BLUE = (135, 206, 235)

class Bird:
    """
    Represents the player-controlled bird in the game.
    Handles bird physics, movement, and rendering.
    """
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.velocity = 0
        self.rect = pygame.Rect(x, y, 30, 30)

    def flap(self):
        """Apply upward force to the bird"""
        self.velocity = FLAP_STRENGTH

    def update(self):
        """Update bird position based on physics"""
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = self.y

    def draw(self, screen):
        """Render the bird on the screen"""
        pygame.draw.rect(screen, BLUE, self.rect)

class Pipe:
    """
    Represents a pipe obstacle in the game.
    Handles pipe movement, collision detection, and rendering.
    """
    def __init__(self, x: int):
        self.x = x
        # Randomly position the gap between pipes
        self.gap_y = random.randint(200, SCREEN_HEIGHT - GROUND_HEIGHT - 200)
        self.top_height = self.gap_y - PIPE_GAP // 2
        self.bottom_y = self.gap_y + PIPE_GAP // 2
        self.top_rect = pygame.Rect(x, 0, 50, self.top_height)
        self.bottom_rect = pygame.Rect(x, self.bottom_y, 50, SCREEN_HEIGHT - self.bottom_y)
        self.passed = False

    def update(self):
        """Move pipe from right to left"""
        self.x -= PIPE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, screen):
        """Render the pipe on the screen"""
        pygame.draw.rect(screen, GREEN, self.top_rect)
        pygame.draw.rect(screen, GREEN, self.bottom_rect)

class Game:
    """
    Main game class that manages the game loop, state, and rendering.
    Implements thread-safe operations for AI integration.
    """
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        self.state_lock = threading.Lock()  # Thread-safe state access
        self.reset_game()

    def reset_game(self):
        """Reset game state to initial values"""
        with self.state_lock:
            self.bird = Bird(SCREEN_WIDTH // 3, SCREEN_HEIGHT // 2)
            self.pipes: List[Pipe] = []
            self.score = 0
            self.game_over = False
            self.last_pipe = pygame.time.get_ticks()

    def handle_events(self):
        """Process user input and window events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_over:
                        self.reset_game()
                    else:
                        with self.state_lock:
                            self.bird.flap()

    def update(self):
        """Update game state including bird, pipes, and collisions"""
        with self.state_lock:
            if self.game_over:
                return

            self.bird.update()

            # Generate new pipes based on time
            current_time = pygame.time.get_ticks()
            if current_time - self.last_pipe > PIPE_FREQUENCY:
                self.pipes.append(Pipe(SCREEN_WIDTH))
                self.last_pipe = current_time

            # Update and check pipes
            for pipe in self.pipes[:]:
                pipe.update()
                
                # Remove off-screen pipes
                if pipe.x < -50:
                    self.pipes.remove(pipe)
                    continue

                # Check for collisions
                if (pipe.top_rect.colliderect(self.bird.rect) or 
                    pipe.bottom_rect.colliderect(self.bird.rect)):
                    self.game_over = True

                # Update score when passing pipes
                if not pipe.passed and pipe.x < self.bird.x:
                    self.score += 1
                    pipe.passed = True

            # Check for ground/ceiling collision
            if self.bird.y < 0 or self.bird.y > SCREEN_HEIGHT - GROUND_HEIGHT:
                self.game_over = True

    def draw(self):
        """Render all game elements"""
        with self.state_lock:
            self.screen.fill(SKY_BLUE)
            
            # Draw game elements
            for pipe in self.pipes:
                pipe.draw(self.screen)
            self.bird.draw(self.screen)
            pygame.draw.rect(self.screen, GREEN, 
                            (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))

            # Draw UI elements
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(score_text, (10, 10))

            if self.game_over:
                game_over_text = font.render("Game Over! Press SPACE to restart", True, WHITE)
                text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                self.screen.blit(game_over_text, text_rect)

            pygame.display.flip()

    def get_state(self) -> dict:
        """
        Return the current game state for the AI agent.
        Provides normalized values for better AI learning.
        """
        with self.state_lock:
            if not self.pipes:
                return {
                    'bird_y': self.bird.y,
                    'bird_velocity': self.bird.velocity,
                    'next_pipe_x': SCREEN_WIDTH,
                    'next_pipe_gap_y': SCREEN_HEIGHT // 2
                }

            next_pipe = min(self.pipes, key=lambda p: p.x if p.x > self.bird.x else float('inf'))
            return {
                'bird_y': self.bird.y,
                'bird_velocity': self.bird.velocity,
                'next_pipe_x': next_pipe.x,
                'next_pipe_gap_y': next_pipe.gap_y
            }

    def run(self):
        """Main game loop"""
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    game = Game()
    game.run() 