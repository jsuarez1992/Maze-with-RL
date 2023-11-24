import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Constants for the game window
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480  # Dimensions of the game window
BACKGROUND_COLOR = (255, 255, 255)  # White background

# Create the game window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Robot Maze Runner")  # Title of the game window

# Constants for the maze
MAZE_ROWS, MAZE_COLUMNS = 10, 10  # Defines the size of the maze (10x10 grid)
CELL_SIZE = 40  # Size of each cell in the maze (pixels)

# Maze structure (1 represents wall, 0 represents path)
maze_structure = [
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
]

# Robot position
robot_position = [1, 0]  # Starting position of the robot (adjusted to a path cell)

# Constants for Reinforcement Learning
num_states = MAZE_ROWS * MAZE_COLUMNS
num_actions = 4  # Up, Down, Left, Right
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
goal_position = [MAZE_ROWS - 1, MAZE_COLUMNS - 1]  # Define the goal position

# Initialize the Q-table
Q_table = np.zeros((num_states, num_actions))

# Functions for Reinforcement Learning
def calculate_reward(current_position, new_position, goal_position):
    # Rewards based on the robot's new position
    if new_position == goal_position:
        return 100  # Highest reward for reaching the goal
    elif maze_structure[new_position[0]][new_position[1]] == 1:
        return -10  # Negative reward for hitting a wall
    else:
        return -1  # Small negative reward for each step

def update_Q_table(current_state, action, reward, next_state):
    # Q-learning formula to update the Q-table
    max_future_q = np.max(Q_table[next_state])
    current_q = Q_table[current_state][action]
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
    Q_table[current_state][action] = new_q

def position_to_state(position):
    # Convert 2D position to a state index in the Q-table
    return position[0] * MAZE_COLUMNS + position[1]

def choose_action(state):
    # Choose the best action based on the current state in the Q-table
    return np.argmax(Q_table[state])

def draw_maze(screen):
    for row in range(MAZE_ROWS):
        for col in range(MAZE_COLUMNS):
            if maze_structure[row][col] == 1:
                color = (0, 0, 0)  # Black for walls
            else:
                color = (255, 255, 255)  # White for paths
            # Draw each cell as a rectangle
            pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def update_robot_position(action):
    global robot_position
    if action == 0:  # Move up
        new_position = [robot_position[0] - 1, robot_position[1]]
        if not is_wall(new_position):
            robot_position = new_position
    elif action == 1:  # Move down
        new_position = [robot_position[0] + 1, robot_position[1]]
        if not is_wall(new_position):
            robot_position = new_position
    elif action == 2:  # Move left
        new_position = [robot_position[0], robot_position[1] - 1]
        if not is_wall(new_position):
            robot_position = new_position
    elif action == 3:  # Move right
        new_position = [robot_position[0], robot_position[1] + 1]
        if not is_wall(new_position):
            robot_position = new_position

def is_wall(position):
    row, col = position
    if row < 0 or col < 0 or row >= MAZE_ROWS or col >= MAZE_COLUMNS:
        return True  # Out of bounds is considered a wall
    return maze_structure[row][col] == 1

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()  # Close the game if the close button is clicked

    # Reinforcement Learning Logic
    current_state = position_to_state(robot_position)
    action = choose_action(current_state)
    update_robot_position(action)  # Update robot's position based on chosen action

    # Calculate reward and update Q-table
    next_state = position_to_state(robot_position)
    reward = calculate_reward(current_state, robot_position, goal_position)
    update_Q_table(current_state, action, reward, next_state)

    # Drawing and display update
    screen.fill(BACKGROUND_COLOR)
    draw_maze(screen)
    # Draw the robot at its current position
    pygame.draw.rect(screen, (0, 0, 255), (robot_position[1] * CELL_SIZE, robot_position[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.update()

    # Add a delay to make the movements visible
    pygame.time.delay(50)  # You can adjust the delay time as needed

    # Check if the robot has reached the goal
    if robot_position == goal_position:
        print("Robot reached the goal!")
        print("Q-table:")
        print(Q_table)
        pygame.quit()
        sys.exit()
