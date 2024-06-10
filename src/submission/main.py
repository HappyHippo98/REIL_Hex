from src.hex_engine import hexPosition
from src.submission.MCTS import MCTS


def mcts_hex_ai(board, action_set):
    hex_game = hexPosition(size=len(board))
    hex_game.board = board
    hex_game.player = -1 if sum(cell for row in board for cell in row) > 0 else 1
    mcts = MCTS(hex_game, iterations=1000)
    best_action = mcts.search(hex_game)
    return best_action


if __name__ == "__main__":
    hex_game = hexPosition(size=4)


    def machine(board, action_set):
        hex_game.board = board
        hex_game.player = -1 if sum(cell for row in board for cell in row) > 0 else 1
        mcts = MCTS(hex_game, iterations=5000)
        return mcts.search(hex_game)


    hex_game.human_vs_machine(machine=machine)
