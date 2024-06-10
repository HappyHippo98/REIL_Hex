import random
import copy
import torch
import numpy as np

from src.submission.MCTSNode import MCTSNode


class MCTS:
    def __init__(self, game, policy_value_net, iterations=1000):
        self.game = game
        self.policy_value_net = policy_value_net
        self.iterations = iterations

    def search(self, initial_state):
        root = MCTSNode(state=initial_state)
        for _ in range(self.iterations):
            node = self._select(root)
            if not self._is_terminal(node.state):
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)
        return root

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
            board_tensor = torch.tensor(state.board).unsqueeze(0).unsqueeze(0).float()
            policy, value = self.policy_value_net(board_tensor)
            action_probs = torch.softmax(policy, dim=1).squeeze().detach().numpy()
            action = self._select_action(state, action_probs)
            state.moove(action)
        return state.winner

    def _select_action(self, state, action_probs):
        actions = state.get_action_space()
        action_probs = action_probs.reshape(state.size, state.size)
        valid_action_probs = np.array([action_probs[action] for action in actions])
        valid_action_probs /= np.sum(valid_action_probs)  # Normalize the probabilities
        action_idx = np.random.choice(len(actions), p=valid_action_probs)
        return actions[action_idx]

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            if node.state.player != result:  # Because the player has already switched after the move
                node.wins += 1
            if node.parent is not None:
                node.parent.increment_visit(node.action)  # Ensure visit count is incremented
            node = node.parent

    def _is_terminal(self, state):
        return state.winner != 0
