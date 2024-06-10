import torch
import torch.optim as optim
from MCTS import MCTS
from src.hex_engine import hexPosition
from src.submission.PolicyValueNetwork import PolicyValueNetwork
import time


def train(policy_value_net, iterations=10, epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(policy_value_net.parameters(), lr=learning_rate)
    mcts = MCTS(game=hexPosition, policy_value_net=policy_value_net, iterations=iterations)

    for epoch in range(epochs):
        state = hexPosition(size=policy_value_net.board_size)
        total_loss = 0
        steps = 0
        wins = 0
        total_visits = 0
        start_time = time.time()

        while state.winner == 0:
            root_node = mcts.search(state)
            best_action = root_node.best_child(exploration_weight=0).action
            state.moove(best_action)

            optimizer.zero_grad()
            board_tensor = torch.tensor(state.board).unsqueeze(0).unsqueeze(0).float()
            policy, value = policy_value_net(board_tensor)
            loss = compute_loss(policy, value, root_node)

            if loss.item() == 0:  # Skip zero loss to avoid unnecessary backpropagation
                print("Zero loss, skipping backpropagation.")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            wins += root_node.wins
            total_visits += root_node.visits

            if steps % 10 == 0:  # Print every 10 steps
                print(f'Epoch: {epoch + 1}, Step: {steps}, Loss: {loss.item():.4f}')

        end_time = time.time()
        epoch_loss = total_loss / steps if steps > 0 else 0
        win_loss_ratio = wins / total_visits if total_visits > 0 else 0
        print(
            f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.4f}, Win/Loss Ratio: {win_loss_ratio:.4f}, Time: {end_time - start_time:.2f}s')


def compute_loss(policy, value, root_node):
    # Generate target policy from MCTS visit counts
    visit_counts = root_node.visit_counts
    total_visits = sum(visit_counts.values())

    if total_visits == 0:
        print("Total visits is zero.")
        return torch.tensor(0.0, requires_grad=True)  # Return zero loss with requires_grad=True

    target_policy = torch.zeros(policy.size(1))  # Initialize target_policy with the correct size
    for action, count in visit_counts.items():
        if action is None:
            print("Warning: action is None in visit_counts.")
            continue
        action_idx = action[0] * root_node.state.size + action[1]  # Assuming action is a tuple (row, col)
        target_policy[action_idx] = count / total_visits

    # Value target
    target_value = torch.tensor([root_node.state.winner], dtype=torch.float32)

    # Print debugging information
    print("Policy:", policy)
    print("Value:", value)
    print("Target Policy:", target_policy)
    print("Target Value:", target_value)

    # Policy loss: Cross-entropy between the predicted policy and the target policy
    policy_loss = torch.nn.functional.cross_entropy(policy, target_policy.unsqueeze(0))

    # Value loss: Mean squared error between the predicted value and the target value
    value_loss = torch.nn.functional.mse_loss(value, target_value)

    return policy_loss + value_loss


if __name__ == "__main__":
    board_size = 4
    policy_value_net = PolicyValueNetwork(board_size=board_size)
    train(policy_value_net)
