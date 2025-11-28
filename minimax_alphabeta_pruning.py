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


# Immediate win or block check
def immediate_win_or_block(board, player, env):
    valid_moves = env.get_valid_moves(board)

    # Check if current player can win immediately
    for move in valid_moves:
        new_board = env.drop_piece(board, move, player)
        if _win_on_board(new_board, player, env):
            return move

    # Check if opponent can win next move; block them
    opponent = -player
    for move in valid_moves:
        new_board = env.drop_piece(board, move, opponent)
        if _win_on_board(new_board, opponent, env):
            return move

    # No immediate win or block
    return None


def avoid_immediate_loss(board, player, env):
    """
    Remove any move that allows the opponent to win immediately next turn.
    Returns a filtered list of safe moves. If all moves lose immediately,
    return all moves (must choose something).
    """

    valid_moves = env.get_valid_moves(board)
    opponent = -player

    safe_moves = []

    for move in valid_moves:
        # Pretend player makes this move
        new_board = env.drop_piece(board, move, player)

        # Now check all opponent replies
        opp_valid = env.get_valid_moves(new_board)
        opponent_can_win = False

        for opp_move in opp_valid:
            opp_board = env.drop_piece(new_board, opp_move, opponent)
            if _win_on_board(opp_board, opponent, env):
                opponent_can_win = True
                break

        if not opponent_can_win:
            safe_moves.append(move)

    # If every move loses immediately, we must pick one – return all
    return safe_moves if safe_moves else valid_moves


# Updated choose_best_move with immediate win/block logic
def choose_best_move(env, depth=5):
    board = env.board.copy()
    player = env.current_player

    # First check immediate win or block
    move = immediate_win_or_block(board, player, env)
    if move is not None:
        return move
    
    safe_moves = avoid_immediate_loss(board, player, env)
    if len(safe_moves) == 1:
        return safe_moves[0]

    best_score = -float("inf")
    best_move = None

    for move in safe_moves:
        new_board = env.drop_piece(board, move, player)
        score, _ = minimax(new_board, depth - 1,
                           -float("inf"), float("inf"),
                           -player, env)
        score = -score

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


if __name__ == "__main__":
    env = Connect4Env(render=True, wait_time=500)
    state, info = env.reset()
    env.render()

    total_reward = {1: 0.0, -1: 0.0}
    total_steps = {1: 0, -1: 0}
    moves = [] 

    while True:
        mover = env.current_player
        action = choose_best_move(env, depth=5)
        print(f"Player {mover} chooses:", action)

        state, reward, terminated, truncated, info = env.step(action)

        # accumulate
        total_reward[mover] += float(reward)
        total_steps[mover] += 1
        moves.append((mover, action, float(reward)))

        env.render()
        print(state)

        if terminated or truncated:
            print("Game over:", info)
            break

    # summary
    def avg(r, n):
        return (r / n) if n > 0 else 0.0

    print("\n=== GAME METRICS ===")
    print(f"Total moves: {total_steps[1] + total_steps[-1]}")
    print(f"Player 1 -> Steps: {total_steps[1]}\nPlayer 2 -> Steps: {total_steps[-1]}")

