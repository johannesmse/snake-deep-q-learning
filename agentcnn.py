import os
# Force torch to use CPU to train
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import config as cfg
from game import Game
from snake import Snake
import torch
import torch.nn as nn
import random
import copy
from collections import deque

class SnakeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN_output_channels = 32
        self.linear_input = 4 + (self.CNN_output_channels * cfg.PIXEL_WIDTH * cfg.PIXEL_HEIGHT)

        # Convolutional part of the neural net
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels = 4,
                out_channels = 32,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels = 32,
                out_channels = self.CNN_output_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU()
        )

        # Fully connected part of the neural net
        self.fully_connected = nn.Sequential(
            nn.Linear(self.linear_input, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 3)
        )
    
    def forward(self, board, direction):
        cnn_out = self.conv(board)
        cnn_out = torch.flatten(cnn_out, start_dim = 1)

        fc_in = torch.cat((cnn_out, direction), dim = 1)
        fc_out = self.fully_connected(fc_in)

        return fc_out


class AgentCNN:
    def __init__(self, game):
        self.game = game
        self.snake = self.game.snake
        self.train_mode = True
        self.rows = int(cfg.GAME_WINDOW_HEIGHT / cfg.SNAKE_SIZE)
        self.columns = int(cfg.GAME_WINDOW_WIDTH / cfg.SNAKE_SIZE)

        self.discount_factor = 0.99
        self.learning_rate = 0.0003
        self.epsilon = 1
        self.epsilon_decay = 0.99998
        self.epsilon_min = 0

        self.replay_buffer = deque(maxlen=50_000)
        self.replay_batch_size = 64
        self.training_freq = 4

        self.generation = 0
        self.steps = 0

        self.model = SnakeNet()
        self.target_model = copy.deepcopy(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()


        self.action_map = {
            "up" : ["left", "right"],
            "down" : ["right", "left"],
            "left" : ["down", "up"],
            "right" : ["up", "down"]
        }

        # Used to encode game state
        self.direction_encoding = {
            "right" : [1, 0, 0, 0],
            "left" : [0, 1, 0, 0],
            "up" : [0, 0, 1, 0],
            "down" : [0, 0, 0, 1]
        }
    def update_snake(self):
        self.snake = self.game.snake

    def get_current_state(self):
        direction = self.direction_encoding[self.snake.direction]
        direction = torch.tensor(direction, dtype=torch.float32)

        # The 4 channels in order: Head, body, tail, food
        board = torch.zeros(4, self.rows, self.columns)

        # Add head
        x, y = self.snake.head
        column = x // cfg.SNAKE_SIZE
        row = y // cfg.SNAKE_SIZE
        if column == 10:
            print("Index out of range")
            print("head:", self.snake.head)
            print("row:", row)
            print("game done:", self.game.done)
            print("food:", self.game.food)

        board[0, row, column] = 1

        # Add body
        for i in range(1, len(self.snake.body) - 1):
            x, y = self.snake.body[i]
            column = x // cfg.SNAKE_SIZE
            row = y // cfg.SNAKE_SIZE

            board[1, row, column] = 1

        # Add tail
        x, y = self.snake.body[-1]
        column = x // cfg.SNAKE_SIZE
        row = y // cfg.SNAKE_SIZE
        
        board[2, row, column] = 1

        # Add food
        x, y = self.game.food
        column = x // cfg.SNAKE_SIZE
        row = y // cfg.SNAKE_SIZE

        board[3, row, column] = 1

        return board, direction
    
    def get_action_epsilon_greedy(self, state):
        """
        Returns an action using an epsilon-greedy policy.
        With probability epsilon, a random action is chosen.
        Otherwise, the greedy action is returned.
        """

        if random.random() > self.epsilon:
            return self.get_action_greedy(state)
        
        return random.randrange(3)

    def get_action_greedy(self, state):
        board, direction = state

        # Convert a single state into a batch of size 1
        board = board.unsqueeze(0)
        direction = direction.unsqueeze(0)

        q_values = self.model(board, direction)

        return torch.argmax(q_values).item()


    def perform_action(self, action):
        """
        Executes the chosen action.

        The agent predicts relative actions:
            0 = turn left
            1 = turn right
            2 = continue straight

        Left and right are interpreted relative to the snake's current direction.
        For example, if the snake is moving right, then:
            0 -> up
            1 -> down
        """

        if action != 2:
            self.snake.add_input(self.model_to_snake_action(action))

    def model_to_snake_action(self, action):
        return self.action_map[self.snake.direction][action]

    def step(self):
        if not self.train_mode:
            self.eval_step()
            return
        
        self.steps += 1
        state = self.get_current_state()

        # Choose and perform action using epsilon greedy policy
        action = self.get_action_epsilon_greedy(state)
        self.perform_action(action)

        # Advance game one step
        self.game.update_game()

        # Get reward and check if game is over
        reward = self.game.reward
        done = self.game.done

        # Get new state if game is not over
        if done:
            next_state = None
            self.generation += 1
        else:
            next_state = self.get_current_state()

        # Add experience to replay buffer
        self.replay_buffer.append((state, action, reward, next_state))

        if self.steps % self.training_freq == 0:
            self.train()

        # Update epsilon to reduce exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network every 2000 steps
        if self.steps % 2000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Print scores in terminal
        if self.steps % 100000 == 0:
            print(f"Steps: {self.steps:,}")
            self.game.print_scores()

    def eval_step(self):
        """
        A separate step method when training is deactivated.
        Plays optimal action and advances game. 
        Does not adjust training variables or add experiences to buffer.
        """

        state = self.get_current_state()
        action = self.get_action_greedy(state)
        self.perform_action(action)
        self.game.update_game()
    
    def train(self):
        """
        Experience replay batch training with Double DQN
        Does two forward passes for online network (current and next states) and one for the target network(next states),
        one backward pass and one Adam update per train() call with buffer of batch_size.
        """

        if len(self.replay_buffer) < self.replay_batch_size:
            return

        batch = random.sample(self.replay_buffer, self.replay_batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Split the state tuples into separate board and direction tuples
        # board_states is a tuple with length replay_batch_size, and each element is a 
        # 3D tensor of shape (4, 10, 10). (4 channels, 10*10 board size)
        board_states, directions = zip(*states)

        # Stack the individual tensors into batched tensors
        # board_states: (batch_size, 4, 10, 10)
        # directions: (batch_size, 4)
        board_states = torch.stack(board_states)
        directions = torch.stack(directions)

        # Remove terminal states because they have no next Q-value
        filtered_next_states = [next_state for next_state in next_states if next_state is not None]

        if filtered_next_states:
            next_board_states, next_directions = zip(*filtered_next_states)
            next_board_states = torch.stack(next_board_states)
            next_directions = torch.stack(next_directions)
    
            # Get q_values for next states from both models to implement Double DQN
            with torch.no_grad():
                next_states_online_q_values = self.model(next_board_states, next_directions)
                next_states_target_q_values = self.target_model(next_board_states, next_directions)

        # Get current state q_values from online network
        q_values = self.model(board_states, directions)
        target_q_values = q_values.clone().detach()

        filtered_next_states_index = 0
        for i, (action, reward, next_state) in enumerate(zip(actions, rewards, next_states)):
            if next_state is None:
                target = reward
            else:
                next_state_max_action = torch.argmax(next_states_online_q_values[filtered_next_states_index]).item()
                target = reward + self.discount_factor * next_states_target_q_values[filtered_next_states_index][next_state_max_action].item()
                filtered_next_states_index += 1

            target_q_values[i][action] = target

        # Clear old gradients
        self.optimizer.zero_grad()

        # Calculate loss and update model
        loss = self.loss_function(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    def save_agent(self):
        torch.save({
        "model_state_dict" : self.model.state_dict(),
        "target_model_state_dict" : self.target_model.state_dict(),
        "optimizer_state_dict" : self.optimizer.state_dict(),
        "steps" : self.steps,
        "epsilon" : self.epsilon,
        "generations" : self.generation
        }, f"agent_checkpoint_{self.steps}.pth")

    def load_agent(self):
        loaded_agent = torch.load("trained-agents/trained_agent.pth")
        self.model.load_state_dict(loaded_agent["model_state_dict"])
        self.target_model.load_state_dict(loaded_agent["target_model_state_dict"])
        self.optimizer.load_state_dict(loaded_agent["optimizer_state_dict"])

        self.steps = loaded_agent["steps"]
        self.epsilon = loaded_agent["epsilon"]
        self.generation = loaded_agent["generations"]

        self.replay_buffer.clear()

    
    