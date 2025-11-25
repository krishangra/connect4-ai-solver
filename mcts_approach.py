import numpy as np
import math
import random
from connect4 import Connect4Env

class MCTSNode:
    """
    A node in the MCTS search tree.
    
    Attributes:
        board: Current board state (numpy array)
        player: Player who will move from this state (1 or -1)
        parent: Parent node reference (None for root)
        children: Dict mapping move -> child MCTSNode
        visits: Number of times this node was visited
        wins: Cumulative score from this node's perspective
        untried_moves: List of moves not yet expanded
    """
    
    def __init__(self, board, player, parent=None, move=None):
        self.board = board.copy()
        self.player = player
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0.0
        self.children = {}
        self.untried_moves = self._get_valid_moves()
    
    def _get_valid_moves(self):
        """Get list of valid moves (non-full columns)."""
        if self._is_terminal():
            return []
        return [c for c in range(7) if self.board[0][c] == 0]
    
    def _is_terminal(self):
        """Check if this is a terminal state (if win or a draw))."""
        return (self._check_winner(1) or 
                self._check_winner(-1) or 
                len([c for c in range(7) if self.board[0][c] == 0]) == 0)
    
    def _check_winner(self, p):
        """Check if player p has won."""
        b = self.board
        rows, cols = 6, 7
        
        for r in range(rows):
            for c in range(cols - 3):
                if np.all(b[r, c:c+4] == p):
                    return True
        for r in range(rows - 3):
            for c in range(cols):
                if np.all(b[r:r+4, c] == p):
                    return True
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(b[r+i][c+i] == p for i in range(4)):
                    return True
        for r in range(3, rows):
            for c in range(cols - 3):
                if all(b[r-i][c+i] == p for i in range(4)):
                    return True
        return False
    
    def is_fully_expanded(self):
        """Check if all moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """Public method to check terminal state."""
        return self._is_terminal()
    
    def get_winner(self):
        """Return winner (1, -1) or None for draw or ongoing."""
        if self._check_winner(1):
            return 1
        if self._check_winner(-1):
            return -1
        return None
    
    def ucb1(self, exploration=1.414):
        """
        Calculate UCB1 score for node selection.
        
        UCB1 = win_rate + C * sqrt(ln(parent_visits) / visits)
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration_term
    
    def best_child(self, exploration=1.414):
        """Select child with highest UCB1 score."""
        return max(self.children.values(), key=lambda c: c.ucb1(exploration))
    
    def most_visited_child(self):
        """Return the child with most visits (best move after search)."""
        return max(self.children.values(), key=lambda c: c.visits)


class MCTS:
    """
    Monte Carlo Tree Search Algorithm.
    
    This algorithm has four phases:
    1. Selection: Traverse tree using UCB1 until reach unexpanded node
    2. Expansion: Add new child node for an untried move
    3. Simulation: Play random moves until game ends
    4. Backpropagation: Update stats and go back up the tree
    """
    
    def __init__(self, num_simulations=1000, exploration=1.414):
        self.num_simulations = num_simulations
        self.exploration = exploration
    
    def search(self, board, player):
        """
        Run MCTS from the given state and return the best move.
        
        Args:
            board: Current board state (6x7 numpy array)
            player: Current player (1 or -1)
            
        Returns:
            Best column to play (0-6)
        """
        # create root node
        root = MCTSNode(board, player)
        
        for _ in range(self.num_simulations):
            node = root
            
            # look for a node that isn't fully expanded
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.best_child(self.exploration)
            
            # if a node isn't terminal and has untried moves then expand the node
            if not node.is_terminal() and node.untried_moves:
                node = self._expand(node)
            
            # rollout
            result = self._simulate(node)
            
            # update everything and then backup
            self._backpropagate(node, result, root.player)
        
        # return the most visited child
        best = root.most_visited_child()
        return best.move
    
    def _expand(self, node):
        """
        Add a new child node.
        
        Picks a random untried move, creates child node, returns it.
        """
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        new_board = self._drop_piece(node.board, move, node.player)
        
        child = MCTSNode(
            board=new_board,
            player=-node.player,
            parent=node,
            move=move
        )
        
        node.children[move] = child
        return child
    
    def _simulate(self, node):
        """
        Random playout until game ends.
        
        Returns the winner (1, -1) or 0 for draw.
        """
        board = node.board.copy()
        player = node.player
        
        winner = self._get_winner(board)
        if winner is not None:
            return winner
        
        while True:
            valid_moves = [c for c in range(7) if board[0][c] == 0]
            
            if not valid_moves:
                return 0
            
            move = random.choice(valid_moves)
            board = self._drop_piece(board, move, player)
            
            winner = self._get_winner(board)
            if winner is not None:
                return winner
            
            player = -player
    
    def _backpropagate(self, node, result, root_player):
        """
        Update stats up the tree.
        
        Args:
            node: Leaf node to start from
            result: Game result (1, -1, or 0)
            root_player: The player MCTS is finding moves for
        """
        while node is not None:
            node.visits += 1
            
            if node.parent is not None:
                if result == -node.player:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += 0.5
            else:
                if result == root_player:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += 0.5
            
            node = node.parent
    
    def _drop_piece(self, board, col, player):
        """Drop a piece in the given column."""
        new_board = board.copy()
        for r in range(5, -1, -1):
            if new_board[r][col] == 0:
                new_board[r][col] = player
                break
        return new_board
    
    def _get_winner(self, board):
        """Check for winner. Returns 1, -1, or None."""
        for p in [1, -1]:
            for r in range(6):
                for c in range(4):
                    if np.all(board[r, c:c+4] == p):
                        return p
            for r in range(3):
                for c in range(7):
                    if np.all(board[r:r+4, c] == p):
                        return p
            for r in range(3):
                for c in range(4):
                    if all(board[r+i][c+i] == p for i in range(4)):
                        return p
            for r in range(3, 6):
                for c in range(4):
                    if all(board[r-i][c+i] == p for i in range(4)):
                        return p
        return None


def choose_best_move(env, num_simulations=1000):
    """
    Choose the best move for the current environment state.
    
    Args:
        env: Connect4Env instance
        num_simulations: Number of MCTS iterations
        
    Returns:
        Best column to play (0-6)
    """
    mcts = MCTS(num_simulations=num_simulations)
    return mcts.search(env.board.copy(), env.current_player)


if __name__ == "__main__":
    
    env = Connect4Env()
    state, info = env.reset()
    
    symbols = {0: '.', 1: 'X', -1: 'O'}
    move_num = 0
    
    while True:
        move_num += 1
        player_symbol = 'X' if env.current_player == 1 else 'O'
        
        print(f"\nMove {move_num} - Player {player_symbol}")
        
        # board
        for row in env.board:
            print(' '.join(symbols[cell] for cell in row))
        print('0 1 2 3 4 5 6')
        
        action = choose_best_move(env, num_simulations=500)
        print(f"MCTS plays column: {action}")
        
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"\nFinal Board:")
            for row in state:
                print(' '.join(symbols[cell] for cell in row))
            print('0 1 2 3 4 5 6')
            
            if 'winner' in info:
                winner = 'X' if info['winner'] == 1 else 'O'
                print(f"\nPlayer {winner} wins")
            else:
                print("\nDraw")
            break