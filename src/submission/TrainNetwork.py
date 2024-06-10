import torch
import torch.optim as optim
from MCTS import MCTS
from src.hex_engine import hexPosition
from src.submission.PolicyValueNetwork import PolicyValueNetwork


def train(policy_value_net, iterations=1000, epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(policy_value_net.parameters(), lr=learning_rate)
    mcts = MCTS(game=hexPosition, policy_value_net=policy_value_net, iterations=iterations)
    for epoch in range(epochs):
        state = hexPosition(size=7)
        while state.winner == 0:
            root_node = mcts.search(state)
            best_action = root_node.best_child(exploration_weight=0).action
            state.moove(best_action)

            optimizer.zero_grad()
            policy, value = policy_value_net(torch.tensor(state.board).unsqueeze(0).unsqueeze(0).float())
            loss = compute_loss(policy, value, root_node)
            loss.backward()
            optimizer.step()

def compute_loss(policy, value, root_node):
    # Generate target policy from MCTS visit counts
    visit_counts = root_node.visit_counts
    total_visits = sum(visit_counts.values())
    target_policy = torch.zeros_like(policy)
    for action, count in visit_counts.items():
        action_idx = action[0] * root_node.state.size + action[1]  # Assuming action is a tuple (row, col)
        target_policy[action_idx] = count / total_visits

    # Value target
    target_value = torch.tensor([root_node.state.winner], dtype=torch.float32)

    # Policy loss: Cross-entropy between the predicted policy and the target policy
    policy_loss = torch.nn.functional.cross_entropy(policy, target_policy)

    # Value loss: Mean squared error between the predicted value and the target value
    value_loss = torch.nn.functional.mse_loss(value, target_value)

    return policy_loss + value_loss

if __name__ == "__main__":
    board_size = 4
    policy_value_net = PolicyValueNetwork(board_size=board_size)
    train(board_size,policy_value_net)
