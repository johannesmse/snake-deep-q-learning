import config as cfg
from snake import Snake
import random
from collections import deque

class Game:
    def __init__(self):
        self.snake = Snake(self.start_position(cfg.GAME_WINDOW_WIDTH), self.start_position(cfg.GAME_WINDOW_HEIGHT), "right")
        self.food = self.preset_food_position()
        self.done = False
        self.score = 0
        self.high_score = 0

        self.last_200_scores = deque(maxlen=200)
        self.average_score = 0
    
    def outside_game_window(self, snake):
        head_x, head_y = snake.head

        return head_x < 0 or head_x > cfg.GAME_WINDOW_WIDTH - cfg.SNAKE_SIZE or head_y < 0 or head_y > cfg.GAME_WINDOW_HEIGHT - cfg.SNAKE_SIZE
    
    def inside_food(self, snake):
        return snake.head == self.food
    
    def handle_food_eaten(self, snake):
        snake.add_body_parts(1)
        self.score += 1
        self.food = self.random_food_position()

    def reset_snake(self):
        #self.snake = Snake(self.start_position(cfg.GAME_WINDOW_WIDTH), self.start_position(cfg.GAME_WINDOW_HEIGHT), "right")
        self.snake = Snake(180, 40, "right")

    def random_food_position(self):
        return [self.random_grid_coordinate(cfg.GAME_WINDOW_WIDTH), self.random_grid_coordinate(cfg.GAME_WINDOW_HEIGHT)]

    def preset_food_position(self):
        center_x = (cfg.GAME_WINDOW_WIDTH // (2 * cfg.SNAKE_SIZE)) * cfg.SNAKE_SIZE
        center_y = (cfg.GAME_WINDOW_HEIGHT // (2 * cfg.SNAKE_SIZE)) * cfg.SNAKE_SIZE
        offset = 2 * cfg.SNAKE_SIZE

        positions = [
            [center_x - offset, center_y - offset],  # Up-left
            [center_x + offset, center_y - offset],  # Up-right
            [center_x - offset, center_y + offset],  # Down-left
            [center_x + offset, center_y + offset]   # Down-right
        ]

        return random.choice(positions)


    def update_game(self):
        if self.done:
            print("Game is finished. Not updating game state.")
            return
        
        # Reward for each step agent takes
        self.reward = -0.05

        # If head is on top of food, respawns food and increments growth queue
        if self.inside_food(self.snake):
            self.handle_food_eaten(self.snake)

        self.snake.move()

        # Checks if snake dies
        if self.outside_game_window(self.snake) or self.snake.inside_itself():
            self.reward = -10
            self.done = True
            self.update_scores()

        # After the snake moved give reward if on top of food
        # Does not respawn the food
        elif self.inside_food(self.snake):
            self.reward = 10
    
    def reset_game(self):
        self.reset_snake()
        self.food = self.preset_food_position()
        self.done = False
        

    def update_scores(self):
        if self.score > self.high_score:
            self.high_score = self.score

        self.last_200_scores.append(self.score)
        self.average_score = sum(self.last_200_scores) / len(self.last_200_scores)
        self.score = 0

    def print_scores(self):
        print(f"Highscore: {self.high_score}")
        print(f"Average Score: {self.average_score:.2f}\n")

    # Helper method to get a random grid coordinate
    def random_grid_coordinate(self, grid_size):
        return cfg.SNAKE_SIZE * random.randint(0, int((grid_size / cfg.SNAKE_SIZE)) - 1)
    
    def start_position(self, grid_size):
        return cfg.SNAKE_SIZE * int((grid_size / cfg.SNAKE_SIZE) / 2)