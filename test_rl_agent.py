from connect4 import Connect4Env
from rl_agent import RLAgent
import random

def play_game():
    env = Connect4Env()
    agent = RLAgent(model_path="rl.pth")

    obs, info = env.reset()
    done = False

    while not done:
        valid_moves = env.get_valid_moves()
        move = agent.select_move(env.board, env.current_player, valid_moves)
        obs, reward, terminated, truncated, info = env.step(move)
        done = terminated or truncated
        print(obs)
        print(reward, info)
        print("========")
play_game()