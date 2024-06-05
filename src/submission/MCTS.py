import torch
import numpy as np
import math


class MCTS:
    def __init__(self, policy_net, value_net, board_size, c_puct=1.0, n_simulations=100):
        self.policy_net = policy_net
        self.value_net = value_net
        self.board_size = board_size
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.Q = {}
        self.N = {}
        self.P = {}

    def search(self, hex_position):
        for _ in range(self.n_simulations):
            if hex_position.winner == 0:
                self._simulate(hex_position)
        return self._get_policy(hex_position)

    def _simulate(self, hex_position):
        state = self._serialize_board(hex_position.board)

        if state not in self.P:
            # Expand the new state
            self.P[state], v = self._expand(hex_position)
            return -v

        if hex_position.winner != 0:
            # If the game is already won, return the value
            print(f"Game already won by player {hex_position.winner}")
            return 0

        max_ucb = -float('inf')
        best_action = None

        # Select the action with the highest UCB
        for action in hex_position.get_action_space():
            ucb = self._get_ucb(state, action)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action

        # Perform the chosen action
        hex_position.moove(best_action)
        print(f"After move {best_action}, board state:")
        hex_position.print()  # Debugging: Print the board state
        if hex_position.winner != 0:
            print(f"Game won by player {hex_position.winner} after move {best_action}")

        v = self._simulate(hex_position)  # Recursively simulate the next move
        hex_position.board[best_action[0]][best_action[1]] = 0  # Unmake the move
        hex_position.player *= -1  # Switch back the player

        self._backup(state, best_action, v)
        return -v

    def _expand(self, hex_position):
        state_tensor = torch.tensor(hex_position.board).float().unsqueeze(0).unsqueeze(0)
        policy, value = self.policy_net(state_tensor)
        policy = policy.detach().numpy().flatten()
        value = value.item()

        self.P[self._serialize_board(hex_position.board)] = policy
        print(f"Expanded node with policy: {policy} and value: {value}")
        return policy, value

    def _get_ucb(self, state, action):
        if (state, action) not in self.Q:
            self.Q[(state, action)] = 0
            self.N[(state, action)] = 0

        total_visits = sum(self.N.get((state, a), 0) for a in self.P[state])
        action_index = action[0] * self.board_size + action[1]

        ucb = self.Q[(state, action)] + self.c_puct * self.P[state][action_index] * math.sqrt(total_visits) / (
                    1 + self.N[(state, action)])
        return ucb

    def _backup(self, state, action, v):
        self.N[(state, action)] = self.N.get((state, action), 0) + 1
        self.Q[(state, action)] = self.Q.get((state, action), 0) + (v - self.Q.get((state, action), 0)) / self.N[
            (state, action)]

    def _get_policy(self, hex_position):
        state = self._serialize_board(hex_position.board)
        policy = np.zeros(hex_position.size * hex_position.size)
        for action in hex_position.get_action_space():
            policy[action[0] * hex_position.size + action[1]] = self.N.get((state, action), 0)
        policy /= np.sum(policy)
        return policy

    def _serialize_board(self, board):
        return tuple(tuple(row) for row in board)