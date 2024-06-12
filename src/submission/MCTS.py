import copy
import random

import numpy as np
import torch

from src.submission.MCTSNode import MCTSNode

class MCTS:
    def __init__(self, game, policy_value_net, iterations=1000, device='cpu'):
        self.game = game
        self.policy_value_net = policy_value_net
        self.iterations = iterations
        self.device = device
        self.root = None  # Persistent root node
        self.node_count = 0

    def search(self, current_state):
        if self.root is None:
            self.root = MCTSNode(state=copy.deepcopy(current_state))  # Initialize root if not set
            current_node = self.root
        else:
            current_node = self._find_or_create_node(current_state)

        for _ in range(self.iterations):
            node = self._select(current_node)
            if not self._is_terminal(node.state):
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)
        return current_node

    def _find_or_create_node(self, state):
        current_node = self.root

        actions = self.get_actions_from_history(state.history)

        for action in actions:

            found = False
            for child in current_node.children:
                if child.action == action:
                    current_node = child
                    found = True
                    break
            if not found:
                print(F"Node has no child with action {action}")
        return current_node

    def get_actions_from_history(self, history):
        actions = []
        history[0] = self.root.state.history[0]
        for idx in range(1, len(history)):
            previous_board = history[idx - 1]
            current_board = history[idx]
            action = self.find_difference(previous_board, current_board)
            actions.append(action)
        return actions

    def find_difference(self, previous_board, current_board):
        for i in range(len(previous_board)):
            for j in range(len(previous_board[i])):
                if previous_board[i][j] != current_board[i][j]:
                    return (i, j)
        return None

    def _select(self, node):
        while not self._is_terminal(node.state) and node.is_fully_expanded():
            node = node.best_child()
        return node

    def _expand(self, node):
        actions = node.state.get_action_space()
        random.shuffle(actions)
        for action in actions:
            action = tuple(action)  # Ensure action is a tuple
            if action not in [child.action for child in node.children]:
                new_state = copy.deepcopy(node.state)
                new_state.moove(action)  # Move the state according to the action
                child_node = MCTSNode(state=new_state, parent=node, action=action)
                self.node_count+=1
                node.add_child(child_node)
                node.visit_counts[action] = 0  # Initialize visit count for the new action
                return child_node
        return node

    def _select_action_with_ucb(self, state, action_probs, visited_actions, exploration_weight=1.4):
        actions = state.get_action_space()
        action_probs = action_probs.reshape(state.size, state.size)

        # Mask the probabilities of already visited actions
        for action in visited_actions:
            action_probs[action[0], action[1]] = 0

        valid_action_probs = np.array([action_probs[action] for action in actions])

        # Ensure valid_action_probs sum to 1
        valid_action_probs /= np.sum(valid_action_probs)

        ucb_values = [
            valid_action_probs[i] + exploration_weight * np.sqrt(
                np.log(np.sum(valid_action_probs) + 1) / (valid_action_probs[i] + 1))
            for i in range(len(actions))
        ]

        max_ucb = max(ucb_values)
        best_actions = [actions[i] for i, ucb in enumerate(ucb_values) if ucb == max_ucb]

        return random.choice(best_actions)

    def _simulate(self, node):
        state = copy.deepcopy(node.state)
        visited_actions = []

        # Add actions from parent nodes to visited_actions
        current_node = node
        while current_node.parent is not None:
            visited_actions.append(current_node.action)
            current_node = current_node.parent

        while state.winner == 0:
            board_tensor = torch.tensor(state.board).unsqueeze(0).unsqueeze(0).float().to(self.device)
            policy, value = self.policy_value_net(board_tensor)
            action_probs = torch.softmax(policy, dim=1).squeeze().detach().cpu().numpy()
            action = self._select_action_with_ucb(state, action_probs, visited_actions)
            visited_actions.append(action)
            state.moove(tuple(action))  # Ensure action is a tuple
        return state.winner

    def _select_action(self, state, action_probs):
        actions = state.get_action_space()
        action_probs = action_probs.reshape(state.size, state.size)
        valid_action_probs = np.array([action_probs[action] for action in actions])
        valid_action_probs /= np.sum(valid_action_probs)
        action_idx = np.random.choice(len(actions), p=valid_action_probs)
        return tuple(actions[action_idx])  # Ensure action is a tuple

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            if node.state.player != result:
                node.wins += 1
            if node.action is not None and node.parent:
                if node.action in node.parent.visit_counts:
                    node.parent.visit_counts[node.action] += 1
                else:
                    node.parent.visit_counts[node.action] = 1
            node = node.parent

    def _is_terminal(self, state):
        return state.winner != 0

    def get_node_count(self):
        return self.node_count
