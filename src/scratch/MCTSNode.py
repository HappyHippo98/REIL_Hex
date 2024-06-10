import math
import random
from copy import deepcopy

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = deepcopy(state)
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.state.get_action_space()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.4):
        choices_weights = [
            (child.wins / child.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        move = self.untried_moves.pop()
        next_state = deepcopy(self.state)
        next_state.moove(move)
        child_node = MCTSNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def tree_policy(self):
        current_node = self
        while not current_node.state.winner:
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def default_policy(self):
        current_simulation = deepcopy(self.state)
        while not current_simulation.winner:
            possible_moves = current_simulation.get_action_space()
            current_simulation.moove(random.choice(possible_moves))
        return current_simulation.winner

    def backup(self, result):
        current_node = self
        while current_node:
            current_node.update(result)
            current_node = current_node.parent

    def get_last_move(self):
        if not self.parent:
            return None
        for r in range(self.state.size):
            for c in range(self.state.size):
                if self.state.board[r][c] != self.parent.state.board[r][c]:
                    return (r, c)
        return None