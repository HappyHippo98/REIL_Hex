import math
import random


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0
        self.depth = parent.depth + 1 if parent else 0  # Set depth based on parent node
        self.visit_counts = {}  # Initialize visit counts as a dictionary

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_action_space())

    def best_child(self, exploration_weight=1.4):
        choices_weights = [
            (child.wins / child.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        max_weight = max(choices_weights)
        best_children = [child for child, weight in zip(self.children, choices_weights) if weight == max_weight]
        return random.choice(best_children)

    def win_probability(self):
        if self.visits == 0:
            return 0.5  # Wenn keine Besuche, nehmen wir 50-50 an
        return self.wins / self.visits