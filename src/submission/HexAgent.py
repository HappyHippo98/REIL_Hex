import torch
import torch.optim as optim
import numpy as np

from src.hex_engine import hexPosition

class HexAgent:
    def __init__(self, board_size, mcts_simulations=100, lr=0.001):
        from src.submission.MCTS import MCTS
        from src.submission.AlphaZeroNet import AlphaZeroNet
        self.board_size = board_size
        self.policy_net = AlphaZeroNet(board_size)
        self.value_net = AlphaZeroNet(board_size)
        self.mcts = MCTS(self.policy_net, self.value_net, board_size, n_simulations=mcts_simulations)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)

    def __call__(self, board, action_set):
        hex_position = hexPosition(size=self.board_size)
        hex_position.board = board
        policy = self.mcts.search(hex_position)
        action_probs = [(action, policy[action[0] * hex_position.size + action[1]]) for action in action_set]
        best_action = max(action_probs, key=lambda x: x[1])[0]
        return best_action

    def self_play(self, n_games=100, max_steps=50):
        training_data = []
        min_steps = self.board_size * 2 - 1  # Minimum number of steps required for a valid game
        for game in range(n_games):
            hex_position = hexPosition(self.board_size)
            hex_position.print()  # Debugging: Drucke das initialisierte Spielfeld
            print(f"Initial winner: {hex_position.winner}")  # Debugging: Drucke den initialen Gewinner
            step = 0
            while hex_position.winner == 0 and step < max_steps:
                policy = self.mcts.search(hex_position)
                action = np.random.choice(len(policy), p=policy)
                print(
                    f"Chosen action: {action}, Policy: {policy}")  # Debugging: Drucke die gewählte Aktion und die Policy
                if hex_position.winner == 0:  # Sicherstellen, dass das Spiel noch nicht gewonnen ist
                    hex_position.moove((action // self.board_size, action % self.board_size))
                    hex_position.print()  # Debugging: Drucke das Spielfeld nach dem Zug
                    print(f"Winner after move: {hex_position.winner}")  # Debugging: Drucke den Gewinner nach dem Zug
                    training_data.append((hex_position.board.copy(), policy.copy(), hex_position.winner))
                    step += 1
                else:
                    print(f"Game ended prematurely. Winner: {hex_position.winner}, Steps: {step}")
            if step < min_steps:
                print(f"Game ended with insufficient steps ({step}). This is invalid.")
            print(f"Game {game + 1}/{n_games} finished, winner: Player {hex_position.winner}, steps: {step}")
        return training_data

    def train(self, training_data, epochs=10, batch_size=32):
        if len(training_data) == 0:
            print("No training data available. Skipping training.")
            return

        self.policy_net.train()
        self.value_net.train()

        for epoch in range(epochs):
            np.random.shuffle(training_data)
            epoch_loss = 0
            num_batches = max(1, len(training_data) // batch_size)  # Sicherstellen, dass es mindestens eine Batch gibt
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                states, policies, values = zip(*batch)
                states = np.array(states)
                policies = np.array(policies)
                values = np.array(values)

                states = torch.tensor(states).float().unsqueeze(1)
                policies = torch.tensor(policies).float()
                values = torch.tensor(values).float().unsqueeze(1)

                self.optimizer.zero_grad()
                pred_policies, pred_values = self.policy_net(states)
                policy_loss = -torch.mean(torch.sum(policies * torch.log(pred_policies), dim=1))
                value_loss = torch.mean((values - pred_values) ** 2)
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs} completed, Average Loss: {avg_loss:.4f}")