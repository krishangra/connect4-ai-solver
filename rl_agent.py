import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from rl_model import Connect4Net
from connect4 import Connect4Env

device = "cuda" if torch.cuda.is_available() else "cpu"
# play against itself to train and learn
def board_to_tensor(board, current_player):
    # -1, 0, 1 (-1 for opponent, 0 for not filled, 1 for current player)
    arr = np.array(board, dtype=np.float32) * float(current_player)
    x=torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return x.to(device)

def epsilon_greedy_nn(state_tensor, valid_moves, epsilon, model):
    # Epsilon greedy algorithm with a cnn instead of a q-table
    if np.random.rand() < epsilon:
        action = int(np.random.choice(valid_moves))
    else:
        # get the greedy action from the neural network
        with torch.no_grad():
            q_values = model(state_tensor)[0].cpu().numpy()
            
        mask = np.full(7, -1e9, dtype=np.float32)
        mask[valid_moves] = 0.0
        q_values = q_values + mask
        action = int(q_values.argmax())
    
    return action

def train_rl_agent(num_episodes, gamma, epsilon, decay_rate):
    env = Connect4Env()
    model = Connect4Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss() #MSE with Q(s,a)
    # Q-learning update like PA 2, but having a problem with the dimension sizes, will work on it.
    # last resort if i can't figure how to use a nn with rl, i will use a q-table like pa2
    #original:
    # Q_table = {}
	# update_count_qsa = {}
	# num_of_actions = len(env.actions)
	
	# def initS(obs):
	# 	s=hash(obs)
	# 	if s not in Q_table:
	# 		Q_table[s]=np.zeros(num_of_actions, dtype=float)
	# 	return s
	
	# for _ in tqdm(range(num_episodes)):
	# 	obs, reward, done, info = env.reset()
	# 	while not done:
	# 		state = initS(obs)
			
	# 		action = epsilon_greedy("train", state, epsilon, Q_table)[0]
			
	# 		# take a step
	# 		next_obs, reward, done, info = env.step(action)
	# 		next_state = initS(next_obs)

	# 		update_count_qsa[(state, action)] = update_count_qsa.get((state, action), 0) + 1 # if the 
	# 		learning_rate = float(1/(1+update_count_qsa[(state,action)]))

	# 		# V_old_opt (s') = max Q_old_opt
	# 		V_old_opt = 0 # if done, it's terminated, there's not 'future' value
	# 		if not done:
	# 			V_old_opt = np.max(Q_table[next_state])
			
	# 		# Q_new_opt
	# 		Q_old_opt = Q_table[state][action]
	# 		Q_table[state][action] = (1-learning_rate)*Q_old_opt + learning_rate*(reward + gamma*V_old_opt)

	# 		obs = next_obs
		
	# 	epsilon = epsilon * decay_rate
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            board = env.board.copy()
            current_player = env.current_player
            valid_moves = env.get_valid_moves()

            state_tensor = board_to_tensor(board, current_player)
            action = epsilon_greedy_nn(state_tensor, valid_moves, epsilon, model)

            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            next_board = env.board.copy()
            next_player = env.current_player

            with torch.no_grad():
                if done:
                    target_q_value = reward # terminated
                else:
                    next_valid = env.get_valid_moves()
                    next_tensor = board_to_tensor(next_board, current_player)
                    q_next = model(next_tensor)[0]

                    # mask invalid next moves
                    next_mask = torch.full_like(q_next, -1e9)
                    next_mask[next_valid] = 0.0
                    q_next = q_next + next_mask

                    target_q_value = reward + gamma * torch.max(q_next).item()
            # current Q(s, :) and target vector
            
            # MSE Loss between current Q(s, ) and target vector
            

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            obs = next_obs
        epsilon = epsilon * decay_rate

    torch.save(model.state_dict(), "rl.pth")
    return model
    

class RLAgent:

    def __init__(self, model_path=None):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.model = Connect4Net().to(self.device)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()

    def select_move(self, board, current_player, valid_moves):
        with torch.no_grad():
            if not isinstance(board, np.ndarray):
                board = np.array(board, dtype=np.int8)
            
            state_tensor = board_to_tensor(board, current_player)

            # q values calculated with CNN instead of like PA2
            # cpu mode for a numpy array, and then flattens (1, 7) into (7,)
            # the output shape was (1,7)--one score per column
            q_values = self.model(state_tensor)[0].cpu().numpy().flatten()

            mask = np.full(7, -1e9, dtype=np.float32) # mask that punishes invalid moves
            mask[valid_moves] = 0.0 # no penalty for valid moves
            
            # valid move = the real neural network score
            # invalid move = an extremely low value (very bad so it will never play an invalid move in real-time)
            masked_q = q_values + mask 
            
            # return the column (1-7) with the highest q-score
            return int(masked_q.argmax())

train_rl_agent(num_episodes=3000, gamma=0.9, epsilon=1, decay_rate=0.999)
        

