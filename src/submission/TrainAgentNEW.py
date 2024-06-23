import time

import numpy as np
from keras.saving.save import load_model

from src.hex_engine import hexPosition
from src.submission.HexAgentNEW import HexAgentNEW


class TrainAgentNew:
    def __init__(self, board_size=8, model_path='current_best_model.h5'):
        self.board_size = board_size
        self.model_path = model_path
        self.model = self.load_or_initialize_model()

    def load_or_initialize_model(self):
        try:
            model = load_model(self.model_path)
            if model.input_shape[1] != self.board_size or model.input_shape[2] != self.board_size:
                raise ValueError("Loaded model has a different board size.")
            print("Loaded existing model.")
        except:
            model = self.build_model()
            model.save(self.model_path)
            print("Initialized new model.")
        return model

    def build_model(self):
        from keras.layers import Input, Conv2D, Activation, Dense, Flatten, Add, BatchNormalization
        from keras.models import Model
        from keras import optimizers
        from keras.regularizers import l2

        cnn_filter_num = 128
        cnn_first_filter_size = 2
        cnn_filter_size = 2
        l2_reg = 0.0001
        res_layer_num = 20
        n_labels = self.board_size * self.board_size
        value_fc_size = 64
        learning_rate = 0.1
        momentum = 0.9

        def _build_residual_block(x, index):
            in_x = x
            res_name = "res" + str(index)
            x = Conv2D(filters=cnn_filter_num, kernel_size=cnn_filter_size, padding="same", data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name=res_name + "_conv1-" + str(cnn_filter_size) + "-" + str(cnn_filter_num))(x)
            x = BatchNormalization(axis=-1, name=res_name + "_batchnorm1")(x)
            x = Activation("relu", name=res_name + "_relu1")(x)
            x = Conv2D(filters=cnn_filter_num, kernel_size=cnn_filter_size, padding="same", data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name=res_name + "_conv2-" + str(cnn_filter_size) + "-" + str(cnn_filter_num))(x)
            x = BatchNormalization(axis=-1, name="res" + str(index) + "_batchnorm2")(x)
            x = Add(name=res_name + "_add")([in_x, x])
            x = Activation("relu", name=res_name + "_relu2")(x)
            return x

        in_x = x = Input((self.board_size, self.board_size, 1))  # NHWC format

        x = Conv2D(filters=cnn_filter_num, kernel_size=cnn_first_filter_size, padding="same", data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name="input_conv-" + str(cnn_first_filter_size) + "-" + str(cnn_filter_num))(x)
        x = BatchNormalization(axis=-1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(res_layer_num):
            x = _build_residual_block(x, i + 1)

        res_out = x

        x = Conv2D(filters=2, kernel_size=1, data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=-1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        policy_out = Dense(n_labels, kernel_regularizer=l2(l2_reg), activation="softmax", name="policy_out")(x)

        x = Conv2D(filters=4, kernel_size=1, data_format="channels_last", use_bias=False, kernel_regularizer=l2(l2_reg), name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=-1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(value_fc_size, kernel_regularizer=l2(l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(l2_reg), activation="tanh", name="value_out")(x)

        model = Model(in_x, [policy_out, value_out], name="hex_model")

        sgd = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        losses = ['categorical_crossentropy', 'mean_squared_error']
        model.compile(loss=losses, optimizer=sgd, metrics=['accuracy', 'mae'])

        model.summary()
        return model

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
        self.model.fit(train_x, train_y, verbose=0, validation_split=0.2, epochs=10, shuffle=True)
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
            print(f'Game #: {i + 1}')
            g = hexPosition(self.board_size)
            player1 = HexAgentNEW(self.model, board_size=self.board_size)
            player2 = HexAgentNEW(self.model, board_size=self.board_size)
            start_time = time.time()
            game, new_training_data = self.play_game(g, player1, player2, False)
            end_time = time.time()
            print(f"Game {i + 1} took: {end_time - start_time:.2f} seconds")
            training_data += new_training_data
        return training_data

if __name__ == "__main__":
    board_size = 4  # You can change this to any size you want
    agent = TrainAgentNew(board_size=board_size)

    for i in range(3):  # Number of iterations, can be increased
        print(f"Starting iteration {i + 1} with board size {board_size}...")
        training_data = agent.selfPlay(5)  # Number of games per iteration
        new_model = agent.trainModel(training_data, i)
        agent.evaluateModel(new_model, agent.model, i)
        agent.model = new_model
        print(f"Completed iteration {i + 1} with board size {board_size}.")