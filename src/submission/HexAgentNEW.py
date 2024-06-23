import numpy as np
from keras.models import load_model
from src.hex_engine import hexPosition

from src.submission.MCTSNEW import MCTSNEW

class HexAgentNEW:
    def __init__(self, model, board_size=8):
        self.model = model
        self.board_size = board_size
        self.MCTS = MCTSNEW(self.model,board_size)

    def getMove(self, game):
        root_node = self.MCTS.expandRoot(game)
        self.MCTS.runSearch(root_node, num_searches=100)
        best_move = max(root_node.children.items(), key=lambda item: item[1].visits)[0]
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
        train_y = {'policy_out': np.array(y_probs).reshape((len(y_probs), self.board_size * self.board_size)), 'value_out': np.array(y_values)}
        return train_x, train_y

    def reshapedSearchProbs(self, search_probs):
        moves = list(search_probs.keys())
        probs = list(search_probs.values())
        reshaped_probs = np.zeros(self.board_size * self.board_size).reshape(self.board_size, self.board_size)
        for move, prob in zip(moves, probs):
            reshaped_probs[move[0]][move[1]] = prob
        return reshaped_probs.reshape(self.board_size * self.board_size)

    def trainModel(self, training_data, iteration):
        train_x, train_y = self.formatTrainingData(training_data)
        np.savez(f'training_data_{iteration}_size_{self.board_size}', train_x, train_y['policy_out'], train_y['value_out'])
        self.model.fit(train_x, train_y, verbose=1, validation_split=0.2, epochs=10, shuffle=True)
        self.model.save(f'new_model_iteration_{iteration}_size_{self.board_size}.h5')
        return self.model

    def evaluateModel(self, new_model, current_model, iteration):
        numEvaluationGames = 40
        newChallengerWins = 0
        threshold = 0.55

        for i in range(int(numEvaluationGames // 2)):
            g = hexPosition(self.board_size)
            game, _ = self.play_game(g, HexAgentNEW(new_model, board_size=self.board_size), HexAgentNEW(current_model, board_size=self.board_size), False)
            if game.winner:
                newChallengerWins += game.winner
        for i in range(int(numEvaluationGames // 2)):
            g = hexPosition(self.board_size)
            game, _ = self.play_game(g, HexAgentNEW(current_model, board_size=self.board_size), HexAgentNEW(new_model, board_size=self.board_size), False)
            if game.winner == -1:
                newChallengerWins += game.winner
        winRate = newChallengerWins / numEvaluationGames
        print('Evaluation win rate: ' + str(winRate))
        with open("evaluation_results.txt", "a") as text_file:
            text_file.write(f"Evaluation results for iteration {iteration}, board size {self.board_size}: {str(winRate)}\n")
        if winRate >= threshold:
            new_model.save(f'current_best_model_size_{self.board_size}.h5')

    def play_game(self, game, player1, player2, show=True):
        new_game_data = []
        while game.winner == 0:
            if show:
                print(game)
            if game.player == 1:
                m = player1.getMove(game)
            else:
                m = player2.getMove(game)
            if m not in game.get_action_space():
                raise Exception("invalid move: " + str(m))
            node = player1.MCTS.visited_nodes[game]
            if game.player == 1:
                search_probs = player1.MCTS.getSearchProbabilities(node)
                board = game.board
            if game.player == -1:
                search_probs = player2.MCTS.getSearchProbabilities(node)
                board = -np.array(game.board).T
            reshaped_search_probs = self.reshapedSearchProbs(search_probs)
            if game.player == -1:
                reshaped_search_probs = reshaped_search_probs.reshape((self.board_size, self.board_size)).T.reshape(self.board_size * self.board_size)

            if np.random.random() > 0.5:
                new_game_data.append((board, reshaped_search_probs, None))
            if np.random.random() > 0.5:
                new_game_data.append((board, reshaped_search_probs, None))
            game.moove(m)
        if show:
            print(game, "\n")

            if game.winner != 0:
                print("player", game.winner, "(", end='')
                print((player1.name if game.winner == 1 else player2.name) + ") wins")
            else:
                print("it's a draw")
        outcome = 1 if game.winner == 1 else 0
        new_training_data = [(board, searchProbs, outcome) for (board, searchProbs, throwaway) in new_game_data]
        return game, new_training_data

    def selfPlay(self, numGames):
        training_data = []
        for i in range(numGames):
            print('Game #: ' + str(i))
            g = hexPosition(self.board_size)
            player1 = HexAgentNEW(self.model, board_size=self.board_size)
            player2 = HexAgentNEW(self.model, board_size=self.board_size)
            game, new_training_data = self.play_game(g, player1, player2, False)
            training_data += new_training_data
        return training_data

if __name__ == "__main__":
    board_size = 8  # You can change this to any size you want
    agent = HexAgentNEW(board_size=board_size)

    for i in range(10):  # Number of iterations, can be increased
        print(f"Starting iteration {i+1} with board size {board_size}...")
        training_data = agent.selfPlay(100)  # Number of games per iteration
        new_model = agent.trainModel(training_data, i)
        agent.evaluateModel(new_model, agent.model, i)
        agent.model = new_model
        print(f"Completed iteration {i+1} with board size {board_size}.")
