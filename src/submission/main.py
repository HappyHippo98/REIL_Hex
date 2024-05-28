import os
import sys
import random
from copy import deepcopy
from src.hex_engine import hexPosition


def main():
    game = hexPosition(size=7)

    while game.winner == 0:
        game.print()

        player = "White (1)" if game.player == 1 else "Black (-1)"
        print(f"{player}'s turn to move.")

        possible_actions = game.get_action_space()

        if not possible_actions:
            break

        chosen_move = random.choice(possible_actions)
        game.moove(chosen_move)
        game.evaluate()

    game.print()

    if game.winner == 1:
        print("White (1) wins!")
    elif game.winner == -1:
        print("Black (-1) wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()
