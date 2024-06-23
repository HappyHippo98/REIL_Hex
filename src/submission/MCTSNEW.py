import random

import numpy as np
import torch
from src.submission.MCTSNodeNEW import MCTSNodeNEW as Node
from src.hex_engine import hexPosition
import copy

class MCTSNEW:
    def __init__(self, model, board_size, UCB_const=2, use_policy=True, use_value=True):
        self.visited_nodes = {}  # maps state to node
        self.model = model
        self.board_size = board_size
        self.UCB_const = UCB_const
        self.use_policy = use_policy
        self.use_value = use_value

    def runSearch(self, root_node, num_searches):
        for _ in range(num_searches):
            selected_node = root_node
            available_moves = selected_node.state.get_action_space()
            while len(available_moves) == len(selected_node.children) and not selected_node.state.winner:
                selected_node = self._select(selected_node)
                available_moves = selected_node.state.get_action_space()
            if not selected_node.state.winner:
                if self.use_policy:
                    if selected_node.state not in self.visited_nodes:
                        selected_node = self.expand(selected_node)
                    outcome = selected_node.value
                    if root_node.state.player == -1:
                        outcome = 1 - outcome
                    self._backprop(selected_node, root_node, outcome)
                else:
                    moves = selected_node.state.get_action_space()
                    np.random.shuffle(moves)
                    for move in moves:
                        new_state = self._copy_state(selected_node.state)
                        new_state.moove(move)
                        if new_state not in self.visited_nodes:
                            break
            else:
                outcome = 1 if selected_node.state.winner == 1 else 0
                self._backprop(selected_node, root_node, outcome)

    def create_children(self, parent_node):
        if parent_node.state.winner == 0 and len(parent_node.state.get_action_space()) != len(parent_node.children):
            for move in parent_node.state.get_action_space():
                next_state = self._copy_state(parent_node.state)  # Use a copy of the state
                next_state.moove(move)
                child_node = Node(next_state, parent_node, parent_node.prior_policy[move[0]][move[1]])
                parent_node.children[move] = child_node

    def _select(self, parent_node):
        children = parent_node.children
        items = children.items()
        if self.use_policy:
            UCB_weights = [(v.UCBWeight(parent_node.visits, self.UCB_const, parent_node.state.player), v) for k, v in items]
        else:
            UCB_weights = [(v.UCBWeight_noPolicy(parent_node.visits, self.UCB_const, parent_node.state.player), v) for k, v in items]
        node = max(UCB_weights, key=lambda c: c[0])
        return node[1]

    def modelPredict(self, state):
        board = np.array(state.board).reshape((1, self.board_size, self.board_size, 1))  # NHWC format
        probs, value = self.model.predict(board,verbose = 0)
        value = value[0][0]
        probs = probs.reshape((self.board_size, self.board_size))
        return probs, value

    def expandRoot(self, state):
        root_node = Node(state, None, 1)
        if self.use_policy or self.use_value:
            probs, value = self.modelPredict(state)
            root_node.prior_policy = probs
        if not self.use_value:
            value = self._simulate(root_node)
        root_node.value = value
        self.visited_nodes[state] = root_node
        self.create_children(root_node)
        return root_node

    def expand(self, selected_node):
        if selected_node.state.winner == 0:
            if self.use_policy or self.use_value:
                probs, value = self.modelPredict(selected_node.state)
                selected_node.prior_policy = probs
            if not self.use_value:
                value = self._simulate(selected_node)
            selected_node.value = value
            self.visited_nodes[selected_node.state] = selected_node
            self.create_children(selected_node)
        return selected_node

    def _simulate(self, next_node):
        state = self._copy_state(next_node.state)  # Use a copy of the state
        while not state.winner:
            available_moves = state.get_action_space()
            index = random.choice(range(len(available_moves)))
            move = available_moves[index]
            state.moove(move)
        return (state.winner + 1) / 2

    def _backprop(self, selected_node, root_node, outcome):
        current_node = selected_node
        if selected_node.state.winner:
            outcome = 1 if selected_node.state.winner == 1 else 0
        while current_node != root_node:
            current_node.updateValue(outcome)
            current_node = current_node.parent
        root_node.updateValue(outcome)

    def getSearchProbabilities(self, root_node):
        children = root_node.children
        items = children.items()
        child_visits = [child.visits for action, child in items]
        sum_visits = sum(child_visits)
        if sum_visits != 0:
            normalized_probs = {action: (child.visits / sum_visits) for action, child in items}
        else:
            normalized_probs = {action: (child.visits / len(child_visits)) for action, child in items}
        return normalized_probs

    def _copy_state(self, state):
        """Create a deep copy of the hexPosition object."""
        new_state = hexPosition(state.size)
        new_state.board = copy.deepcopy(state.board)
        new_state.player = state.player
        new_state.winner = state.winner
        new_state.history = copy.deepcopy(state.history)
        return new_state
