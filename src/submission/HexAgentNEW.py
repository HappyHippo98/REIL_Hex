import time
import numpy as np
from keras.models import load_model
from src.hex_engine import hexPosition
from src.submission.MCTSNEW import MCTSNEW


class HexAgentNEW:
    def __init__(self, model, board_size=8, use_mcts=True):
        self.model = model
        self.board_size = board_size
        self.use_mcts = use_mcts
        if use_mcts:
            self.MCTS = MCTSNEW(self.model, board_size)

    def getMove(self, game):
        if self.use_mcts:
            root_node = self.MCTS.expandRoot(game)
            self.MCTS.runSearch(root_node, num_searches=100)
            best_move = max(root_node.children.items(), key=lambda item: item[1].visits)[0]
        else:
            board_input = np.array(game.board).reshape(1, self.board_size, self.board_size, 1)
            policy, value = self.model.predict(board_input)
            policy = policy.reshape(self.board_size, self.board_size)

            # Flatten the policy and sort indices by probability
            sorted_moves = np.argsort(policy.flatten())[::-1]
            for move in sorted_moves:
                row, col = divmod(move, self.board_size)
                if game.board[row][col] == 0:
                    best_move = (row, col)
                    break
        return best_move

    def formatTrainingData(self, training_data):
        x = []
        y_values = []
        y_probs = []
        for (board, probs, value) in training_data:
            x.append(board)
            y_probs.append(probs)
            y_values.append(value)

        train_x = np.array(x).reshape((len(x), self.board_size, self.board_size, 1))  # NHWC format
        train_y = {'policy_out': np.array(y_probs).reshape((len(y_probs), self.board_size * self.board_size)),
                   'value_out': np.array(y_values)}
        return train_x, train_y

    def reshapedSearchProbs(self, search_probs):
        moves = list(search_probs.keys())
        probs = list(search_probs.values())
        reshaped_probs = np.zeros(self.board_size * self.board_size).reshape(self.board_size, self.board_size)
        for move, prob in zip(moves, probs):
            reshaped_probs[move[0]][move[1]] = prob
        return reshaped_probs.reshape(self.board_size * self.board_size)


    def play_vs_human(self):
        game = hexPosition(self.board_size)
        human_player = -1

        def ai_move_function(board, action_set):
            game.board = board
            move = self.getMove(game)
            return move

        game.human_vs_machine(human_player=human_player, machine=ai_move_function)

if __name__ == "__main__":
    import os
    import logging

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger().setLevel(logging.WARNING)

    board_size = 7  # You can change this to any size you want
    model_path = 'current_best_model.h5'  # Path to the trained model
    model = load_model(model_path)
    agent = HexAgentNEW(model, board_size=board_size,use_mcts=False)

    agent.play_vs_human()
