import random
import copy
from .MCTSNode import MCTSNode

class MCTS:
    def __init__(self, game, iterations=1000):
        self.game = game
        self.iterations = iterations

    def search(self, initial_state):
        root = MCTSNode(state=initial_state)
        for _ in range(self.iterations):
            node = self._select(root)
            if not self._is_terminal(node.state):
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)
        return root.best_child(exploration_weight=0).action

    def _select(self, node):
        while not self._is_terminal(node.state) and node.is_fully_expanded():
            node = node.best_child()
        return node

    def _expand(self, node):
        actions = node.state.get_action_space()
        for action in actions:
            if action not in [child.action for child in node.children]:
                new_state = copy.deepcopy(node.state)
                new_state.moove(action)
                child_node = MCTSNode(state=new_state, parent=node, action=action)
                node.add_child(child_node)
                return child_node
        return node

    def _simulate(self, node):
        state = copy.deepcopy(node.state)
        while state.winner == 0:
            action = random.choice(state.get_action_space())
            state.moove(action)
        return state.winner

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            if node.state.player != result:  # Weil der Spieler nach dem Zug bereits gewechselt hat
                node.wins += 1
            node = node.parent

    def _is_terminal(self, state):
        return state.winner != 0
