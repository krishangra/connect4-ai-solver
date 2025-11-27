import gymnasium as gym
import numpy as np
import pygame

# CITATION: ChatGPT used here to assist in creating Connect4 Environment
class Connect4Env(gym.Env):
    """Gymnasium-compatible Connect 4 environment."""

    def __init__(self, render=False):
        super().__init__()

        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)

        self.render_mode = render
        self.pygame_initialized = False
        self.rows = 6
        self.cols = 7

        # 0 = empty, 1 = player, -1 = opponent
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.rows, self.cols), dtype=np.int8
        )

        self.action_space = gym.spaces.Discrete(self.cols)

        self.board = None
        self.current_player = 1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        if self.render_mode:
            self.render()
        return self.board.copy(), {}

    def step(self, action):
        # Validate action
        if action < 0 or action >= self.cols or self.board[0][action] != 0:
            if self.render_mode:
                self.render()
            # Illegal move → negative reward + terminate episode
            return self.board.copy(), -10.0, True, False, {"illegal_move": True}

        # Drop piece in column
        for r in reversed(range(self.rows)):
            if self.board[r][action] == 0:
                self.board[r][action] = self.current_player
                break

        # Check win
        if self._check_win(self.current_player):
            if self.render_mode:
                self.render()
            return self.board.copy(), 1.0, True, False, {"winner": self.current_player}

        # Check draw
        if not (self.board == 0).any():
            if self.render_mode:
                self.render()
            return self.board.copy(), 0.0, True, False, {"draw": True}

        # Switch player (but we don't simulate opponent moves here)
        self.current_player *= -1

        if self.render_mode:
            self.render()

        return self.board.copy(), 0.0, False, False, {}

    # Win detection
    def _check_win(self, p):
        b = self.board

        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if np.all(b[r, c:c+4] == p):
                    return True

        # Vertical
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if np.all(b[r:r+4, c] == p):
                    return True

        # Diagonal down-right
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(b[r+i][c+i] == p for i in range(4)):
                    return True

        # Diagonal up-right
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(b[r-i][c+i] == p for i in range(4)):
                    return True

        return False
    
    # get valid moves at the current state
    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[0][c] == 0]
    

    def render(self):
        if not self.render_mode:
            return 

        if not self.pygame_initialized:
            pygame.init()

            self.square_size = 80
            self.width = self.cols * self.square_size
            self.height = (self.rows + 1) * self.square_size

            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Connect 4")

            self.font = pygame.font.SysFont("Times New Roman", 24)
            self.pygame_initialized = True

        # Draw background
        self.screen.fill(self.BLUE)

        # Draw board slots
        for r in range(self.rows):
            for c in range(self.cols):

                # Blue board rectangle
                pygame.draw.rect(
                    self.screen,
                    self.BLUE,
                    (c * self.square_size,
                     (r + 1) * self.square_size,
                     self.square_size,
                     self.square_size)
                )

                # Piece
                piece = self.board[r][c]
                color = self.BLACK
                if piece == 1:
                    color = self.RED
                elif piece == -1:
                    color = self.YELLOW

                pygame.draw.circle(
                    self.screen,
                    color,
                    (
                        int(c * self.square_size + self.square_size / 2),
                        int((r + 1) * self.square_size + self.square_size / 2)
                    ),
                    self.square_size // 2 - 5
                )

        pygame.display.update()

# Example usage
if __name__ == "__main__":
    env = Connect4Env(render=True)
    s, _ = env.reset()

    pygame.time.wait(5000)

    done = False
    while not done:
        action = np.random.choice(env.get_valid_moves())
        s, r, done, _, info = env.step(action)
        pygame.time.wait(500)
