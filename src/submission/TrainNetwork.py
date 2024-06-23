import copy
import random

import torch
import torch.optim as optim
import os
from MCTS import MCTS
from src.hex_engine import hexPosition
from src.submission.MCTSNode import MCTSNode
from src.submission.PolicyValueNetwork import PolicyValueNetwork
import time
import matplotlib.pyplot as plt


class TrainNetwork:
    def __init__(self, policy_value_net, iterations=100, epochs=1000, learning_rate=0.00005, device=torch.device("cpu")):
        self.policy_value_net = policy_value_net
        self.iterations = iterations
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.losses = []
        self.win_loss_ratios = []
        self.times = []

        # Create the NetData directory if it does not exist
        os.makedirs("NetData", exist_ok=True)

    def train(self):
        self.policy_value_net.to(self.device)
        optimizer = optim.Adam(self.policy_value_net.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            game_board = hexPosition(size=self.policy_value_net.board_size)
            mcts = MCTS(game=hexPosition, policy_value_net=self.policy_value_net, iterations=self.iterations,
                        device=self.device)

            total_loss = 0
            steps = 0
            wins = 0
            total_visits = 0
            start_time = time.time()

            while game_board.winner == 0:
                if game_board.player == -1: # enemy move
                    chosen_random_action = random.choice(game_board.get_action_space())
                    parent = mcts.find_or_create_node(game_board)
                    game_board.moove(chosen_random_action)
                    newMctsNode = MCTSNode(state=game_board, parent=parent, action=chosen_random_action)
                    mcts.node_count += 1
                    parent.add_child(newMctsNode)
                    parent.visit_counts[chosen_random_action] = 0
                    print(game_board.get_action_space())
                    continue
                root_node = mcts.search(game_board) # AI MOVE
                best_action = root_node.best_child(exploration_weight=0).action
                game_board.moove(best_action)

                optimizer.zero_grad()
                board_tensor = torch.tensor(game_board.board).unsqueeze(0).unsqueeze(0).float().to(self.device)
                policy, value = self.policy_value_net(board_tensor)
                loss = self.compute_loss(policy, value, root_node)

                if loss.item() == 0:
                    print("Zero loss, skipping backpropagation.")
                    continue

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1
                wins += root_node.wins
                total_visits += root_node.visits

            end_time = time.time()
            epoch_loss = total_loss / steps if steps > 0 else 0
            win_loss_ratio = wins / total_visits if total_visits > 0 else 0

            self.losses.append(epoch_loss)
            self.win_loss_ratios.append(win_loss_ratio)
            self.times.append(end_time - start_time)

            print(f'Epoch: {epoch + 1}, Step: {steps}, Loss: {loss.item():.4f}')
            print(
                f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.4f}, Win/Loss Ratio: {win_loss_ratio:.4f}, Time: {end_time - start_time:.2f}s')

        self.visualize_results()
        torch.save(self.policy_value_net.state_dict(), 'NetData/net.pth')

    def compute_loss(self, policy, value, root_node):
        visit_counts = root_node.visit_counts
        total_visits = sum(visit_counts.values())

        target_policy = torch.zeros(policy.size(1), device=self.device)
        for action, count in visit_counts.items():
            if action is None:
                print("Warning: action is None in visit_counts.")
                continue
            action_idx = action[0] * root_node.state.size + action[1]
            target_policy[action_idx] = count / total_visits

        win_prob = root_node.wins / root_node.visits if root_node.visits > 0 else 0

        if root_node.state.player == 1:
            target_value = win_prob
        else:
            target_value = 1 - win_prob

        if target_value > 0:
            target_value = 1
        if target_value < 0:
            target_value = -1

        target_value = torch.tensor([target_value], dtype=torch.float32, device=self.device)
        policy_loss = torch.nn.functional.cross_entropy(policy, target_policy.unsqueeze(0))
        value_loss = torch.nn.functional.mse_loss(value, target_value.view_as(value))

        return policy_loss + value_loss

    def visualize_results(self):
        epochs = range(1, self.epochs + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.win_loss_ratios, label='Win/Loss Ratio')
        plt.xlabel('Epochs')
        plt.ylabel('Win/Loss Ratio')
        plt.title('Win/Loss Ratio')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def load_policy_value_net(model_path, board_size, device=torch.device("cpu")):
        model = PolicyValueNetwork(board_size=board_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        return model


if __name__ == "__main__":
    board_size = 4
    generateNewNet = True

    if generateNewNet:
        policy_value_net = PolicyValueNetwork(board_size=board_size)
    else:
        policy_value_net = TrainNetwork.load_policy_value_net('NetData/net.pth', board_size)

    if torch.cuda.is_available():
        print("Running on GPU:")
        start_gpu = time.time()
        trainer = TrainNetwork(policy_value_net, device=torch.device("cuda"))
        trainer.train()
        end_gpu = time.time()
        print(f"Time taken on GPU: {end_gpu - start_gpu:.2f}s")
    else:
        print("CUDA is not available. Running on CPU.")
        start_cpu = time.time()
        trainer = TrainNetwork(policy_value_net, device=torch.device("cpu"))
        trainer.train()
        end_cpu = time.time()
        print(f"Time taken on CPU: {end_cpu - start_cpu:.2f}s")
