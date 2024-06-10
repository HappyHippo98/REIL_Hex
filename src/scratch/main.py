from src.hex_engine import hexPosition
from src.scratch.MCTS import MCTS


def hex_ai_game(state, mcts, human_player=1, simulations_number=1000):
    while state.winner == 0:
        state.print()
        if state.player == human_player:
            # Human move
            move = input("Enter your move (e.g., 'A1'): ")
            coordinates = translator(move)
            state.moove(coordinates)
        else:
            # MCTS move
            best_node = mcts.best_action(simulations_number)
            best_move = best_node.get_last_move()
            state.moove(best_move)
        # Update MCTS with the new move
        mcts.move_and_update(state)
    state.print()
    if state.winner == human_player:
        print("Human wins!")
    else:
        print("MCTS wins!")

    # Display MCTS metrics
    print(f"MCTS win/loss ratio: {mcts.get_win_loss_ratio():.2f}")
    print(f"Tree size: {mcts.get_tree_size()}")
    print(f"Total visits: {mcts.root.visits}")
    print(f"Total wins: {mcts.root.wins}")

def translator(move):
    letter, number = move[0], int(move[1:])
    row = number - 1
    col = ord(letter.upper()) - ord('A')
    return (row, col)

if __name__ == "__main__":
    initial_state = hexPosition(size=4)
    mcts = MCTS(initial_state)

    # Increase the number of simulations
    simulations_number = 10000

    # Train the MCTS agent for the first move
    mcts.best_action(simulations_number)

    # Play against the trained MCTS agent
    hex_ai_game(initial_state, mcts, human_player=1, simulations_number=simulations_number)