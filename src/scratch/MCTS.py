from copy import deepcopy

from src.scratch.MCTSNode import MCTSNode


class MCTS:
    def __init__(self, state):
        self.root = MCTSNode(state)

    def best_action(self, simulations_number):
        for _ in range(simulations_number):
            leaf = self.root.tree_policy()
            simulation_result = leaf.default_policy()
            leaf.backup(simulation_result)
        return self.root.best_child(exploration_weight=0)

    def move_and_update(self, move):
        for child in self.root.children:
            if child.state.board == self.root.state.board:
                self.root = child
                self.root.parent = None
                return
        new_state = deepcopy(self.root.state)
        new_state.moove(move)
        self.root = MCTSNode(new_state)

    def get_win_loss_ratio(self):
        return self.root.wins / self.root.visits if self.root.visits > 0 else 0

    def get_tree_size(self, node=None):
        if node is None:
            node = self.root
        if not node.children:
            return 1
        return 1 + sum(self.get_tree_size(child) for child in node.children)