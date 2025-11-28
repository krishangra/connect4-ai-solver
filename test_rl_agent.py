from connect4 import Connect4Env
from rl_agent import RLAgent, choose_smart_move, train_rl_agent
import random
from tqdm import tqdm
import pickle

def evaluate_agent(filename, episodes=10000):
    env = Connect4Env()
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            Q_table = obj
    except Exception:
        print("File doesn't exist")
    
    agent = RLAgent(Q_table)
    wins = 0
    losses = 0
    draws = 0
    for episode in tqdm(range(episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            valid_moves = env.get_valid_moves()

            if env.current_player == 1:
                action = agent.select_move(env.board, env.current_player, valid_moves)
            else:
                action = choose_smart_move(env.board, valid_moves, env.current_player)

            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                if "winner" in info and info["winner"] == 1:
                    wins += 1
                elif  "winner" in info and info["winner"] == -1:
                    losses += 1
                elif info.get("draw"):
                    draws += 1
                done = True

    print(f"Over {episodes} games vs random:")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"% Win Rate: {round(float(wins/(wins+losses+draws))*100, 2)}%")

num_episodes=100000
decay_rate=0.99999
Q_table = train_rl_agent(num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate)
with open('Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
filename = 'Q_table_'+str(num_episodes)+'_'+str(decay_rate)+'.pickle'
evaluate_agent(filename, episodes=10000)