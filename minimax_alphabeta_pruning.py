import numpy as np
import gymnasium as gym
from connect4 import Connect4Env


# Heuristic evaluation function
def evaluate_position(board, player):
    """
    Simple evaluation scoring:
      +5 for each open 3-in-a-row,
      +2 for open 2-in-a-row,
      and symmetric negative values for opponent.
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

    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            total += score_line(board[r, c:c+4])

    # Vertical
    for r in range(rows - 3):
        for c in range(cols):
            total += score_line(board[r:r+4, c])

    # Diagonal down-right
    for r in range(rows - 3):
        for c in range(cols - 3):
            total += score_line(np.array([board[r+i][c+i] for i in range(4)]))

    # Diagonal up-right
    for r in range(3, rows):
        for c in range(cols - 3):
            total += score_line(np.array([board[r-i][c+i] for i in range(4)]))

    return total


# Minimax (Negamax) with Alpha-Beta Pruning
def minimax(board, depth, alpha, beta, player, env):
    """
    Negamax minimax.
    player = 1 or -1
    Returns (score, best_move)
    """

    valid_moves = env.get_valid_moves(board)

    # Terminal search depth or no moves available
    if depth == 0 or not valid_moves:
        return evaluate_position(board, player), None

    # Immediate win check
    for move in valid_moves:
        new_board = env.drop_piece(board, move, player)
        if _win_on_board(new_board, player, env):
            return 10**6, move

    best_score = -float("inf")
    best_move = None

    for move in valid_moves:
        new_board = env.drop_piece(board, move, player)

        # Opponent search (negamax)
        score, _ = minimax(new_board, depth - 1, -beta, -alpha, -player, env)
        score = -score

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            break  # prune branch

    return best_score, best_move


# Check win on a hypothetical board without breaking environment state
def _win_on_board(board, player, env):
    saved = env.board
    env.board = board
    result = env._check_win(player)
    env.board = saved
    return result


# Choose move entry point
def choose_best_move(env, depth=5):
    board = env.board.copy()
    player = env.current_player
    score, move = minimax(board, depth, -float("inf"), float("inf"), player, env)
    return move


env = Connect4Env(render=True, wait_time=500)
state, info = env.reset()
env.render()

while True:
    action = choose_best_move(env, depth=5)
    print("AI chooses:", action)

    state, reward, terminated, truncated, info = env.step(action)
    print(state)

    if terminated:
        print("Game over:", info)
        break
