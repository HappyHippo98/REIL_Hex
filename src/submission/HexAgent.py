import torch
import numpy as np
from src.hex_engine import hexPosition
from src.submission.PolicyValueNetwork import PolicyValueNetwork


class HexAgent:
    def __init__(self, model_path, board_size, device=torch.device("cpu")):
        self.device = device
        self.policy_value_net = self.load_trained_model(model_path, board_size, device)

    def load_trained_model(self, model_path, board_size, device):
        model = PolicyValueNetwork(board_size=board_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def get_best_move(self, board):
        board_tensor = torch.tensor(board).unsqueeze(0).unsqueeze(0).float().to(self.device)
        policy, _ = self.policy_value_net(board_tensor)
        action_probs = torch.softmax(policy, dim=1).squeeze().detach().cpu().numpy()
        action_probs = action_probs.reshape(len(board), len(board))

        # Mask invalid moves
        hex_game = hexPosition(size=len(board))
        hex_game.board = board
        valid_moves = np.array(hex_game.get_action_space())
        move_probs = np.array([action_probs[move[0], move[1]] for move in valid_moves])
        best_move_idx = np.argmax(move_probs)
        best_move = valid_moves[best_move_idx]
        return tuple(best_move)

    def select_action(self, board, action_set):
        return self.get_best_move(board)


if __name__ == "__main__":
    board_size = 4
    model_path = 'NetData/net.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hex_game = hexPosition(size=board_size)
    agent = HexAgent(model_path, board_size, device)

    hex_game.human_vs_machine(human_player=1, machine=agent.select_action)
