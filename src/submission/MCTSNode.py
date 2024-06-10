import math

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0
        self.visit_counts = {}  # Initialize visit counts

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_action_space())

    def best_child(self, exploration_weight=1.4):
        choices_weights = [
            (child.wins / child.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def add_child(self, child_node):
        self.children.append(child_node)
        if child_node.action is not None:
            self.visit_counts[child_node.action] = 0  # Initialize visit count for the action

    def increment_visit(self, action):
        if action is not None:
            if action in self.visit_counts:
                self.visit_counts[action] += 1
            else:
                self.visit_counts[action] = 1  # Initialize if not present
