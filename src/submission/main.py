from src.submission.HexAgentNEW import HexAgentNEW

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
