from src.hex_engine import hexPosition
from src.submission.HexAgent import HexAgent

if __name__ == "__main__":
    board_size = 4
    agent = HexAgent(board_size, mcts_simulations=10000, lr=0.001)

    for iteration in range(20):
        print(f"Iteration {iteration + 1} started")
        training_data = agent.self_play(n_games=100)
        agent.train(training_data, epochs=1000, batch_size=32)
        print(f"Iteration {iteration + 1} completed")

    def trained_agent(board, action_set):
        return agent(board, action_set)

    hex_position = hexPosition(board_size)
    hex_position.human_vs_machine(human_player=1, machine=trained_agent)