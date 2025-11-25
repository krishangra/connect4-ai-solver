import gymnasium as gym
import numpy as np

class Connect4Env(gym.Env):
    """Simple, modern Gymnasium-compatible Connect 4 environment."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        self.rows = 6
        self.cols = 7

        # 0 = empty, 1 = player, -1 = opponent
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.rows, self.cols), dtype=np.int8
        )

        self.action_space = gym.spaces.Discrete(self.cols)

        self.board = None
        self.current_player = 1  # agent always starts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        return self.board.copy(), {}

    def step(self, action):
        # Validate action
        if action < 0 or action >= self.cols or self.board[0][action] != 0:
            # Illegal move → negative reward + terminate episode
            return self.board.copy(), -10.0, True, False, {"illegal_move": True}

        # Drop piece in column
        for r in reversed(range(self.rows)):
            if self.board[r][action] == 0:
                self.board[r][action] = self.current_player
                break

        # Check win
        if self._check_win(self.current_player):
            return self.board.copy(), 1.0, True, False, {"winner": self.current_player}

        # Check draw
        if not (self.board == 0).any():
            return self.board.copy(), 0.0, True, False, {"draw": True}

        # Switch player (but we don't simulate opponent moves here)
        self.current_player *= -1

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

# Example usage
if __name__ == "__main__":
    env = Connect4Env()

    state, info = env.reset()
    print("Initial Board:")
    print(state)

    next_state, reward, terminated, truncated, info = env.step(3)

    print("\nAfter Action 3:")
    print(next_state)
    print("Reward:", reward)
    print("Terminated?", terminated)
