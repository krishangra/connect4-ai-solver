import numpy as np

# Utility Functions
def get_valid_moves(board):
    return [c for c in range(board.shape[1]) if board[0][c] == 0]


def drop_piece(board, col, player):
    new_board = board.copy()
    rows = board.shape[0]
    for r in reversed(range(rows)):
        if new_board[r][col] == 0:
            new_board[r][col] = player
            return new_board
    return new_board


# Heuristic function
def evaluate_position(board, player):
    """
    Simple evaluation:
      +5 for each open 3-in-a-row,
      +2 for open 2-in-a-row,
      and the opposite for opponent.
    """
    def score_line(line):
        score = 0
        if np.count_nonzero(line == player) == 3 and np.count_nonzero(line == 0) == 1:
            score += 5
        if np.count_nonzero(line == player) == 2 and np.count_nonzero(line == 0) == 2:
            score += 2

        opp = -player
        if np.count_nonzero(line == opp) == 3 and np.count_nonzero(line == 0) == 1:
            score -= 5
        if np.count_nonzero(line == opp) == 2 and np.count_nonzero(line == 0) == 2:
            score -= 2

        return score

    rows, cols = board.shape
    total = 0

    # horizontal
    for r in range(rows):
        for c in range(cols - 3):
            total += score_line(board[r, c:c+4])

    # vertical
    for r in range(rows - 3):
        for c in range(cols):
            total += score_line(board[r:r+4, c])

    # diag down-right
    for r in range(rows - 3):
        for c in range(cols - 3):
            total += score_line(np.array([board[r+i][c+i] for i in range(4)]))

    # diag up-right
    for r in range(3, rows):
        for c in range(cols - 3):
            total += score_line(np.array([board[r-i][c+i] for i in range(4)]))

    return total


def check_win(board, player, env):
    saved = env.board
    env.board = board
    result = env._check_win(player)
    env.board = saved
    return result


# Minimax algorithm with alpha-beta pruning
def minimax(board, depth, alpha, beta, player, env):
    """
    Negamax minimax with alpha-beta pruning.
    player =  1 or -1  (current player to move)
    Returns: (score, chosen_move)
    """

    valid_moves = get_valid_moves(board)

    # no moves left
    if depth == 0 or len(valid_moves) == 0:
        return evaluate_position(board, player), None

    # Check immediate win
    for move in valid_moves:
        new_board = drop_piece(board, move, player)
        if check_win(new_board, player, env):
            return (10**6, move)

    best_score = -float("inf")
    best_move = None

    for move in valid_moves:
        new_board = drop_piece(board, move, player)

        # Opponent move: score is negated (negamax)
        score, _ = minimax(new_board, depth - 1, -beta, -alpha, -player, env)
        score = -score

        if score > best_score:
            best_score = score
            best_move = move

        # pruning
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best_score, best_move


def choose_best_move(env, depth=5):
    board = env.board.copy()
    player = env.current_player
    score, move = minimax(board, depth, -float("inf"), float("inf"), player, env)
    return move


import gymnasium as gym
from connect4 import Connect4Env


env = Connect4Env()
state, info = env.reset()

while True:
    # Minimax chooses action
    action = choose_best_move(env, depth=5)
    print("AI chooses:", action)

    state, reward, terminated, truncated, info = env.step(action)
    print(state)

    if terminated:
        print("Game over:", info)
        break
