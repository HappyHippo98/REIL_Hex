import math


class MCTSNodeNEW:
    def __init__(self, state, parent_node, prior_prob):
        self.state = state
        self.children = {}
        self.visits = 0
        self.value = 0.5
        self.prior_prob = prior_prob
        self.visits = 0
        self.parent = parent_node

    def updateValue(self, outcome, debug=False):
        self.value = (self.visits * self.value + outcome) / (self.visits + 1)
        self.visits += 1

    def UCBWeight_noPolicy(self, parent_visits, UCB_const, player):
        if player == -1:
            return (1 - self.value) + UCB_const * math.sqrt(parent_visits) / (1 + self.visits)
        else:
            return self.value + UCB_const * math.sqrt(parent_visits) / (1 + self.visits)

    def UCBWeight(self, parent_visits, UCB_const, player):
        if player == -1:
            return (1 - self.value) + UCB_const * self.prior_prob / (1 + self.visits)
        else:
            return self.value + UCB_const * self.prior_prob / (1 + self.visits)