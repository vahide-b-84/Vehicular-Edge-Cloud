# DQN_template.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu'):
        super(DQNNetwork, self).__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            prev_dim = h

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)

class DQNAgent:
    def __init__(self, num_states, num_actions, hidden_layers, device='cpu', gamma=0.99, lr=1e-3, tau=0.005, buffer_size=100000, batch_size=64, activation='relu'):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.policy_net = DQNNetwork(num_states, num_actions, hidden_layers, activation).to(device)
        self.target_net = DQNNetwork(num_states, num_actions, hidden_layers, activation).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = []
        self.buffer_size = buffer_size

    '''def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()'''
    
    def select_action(self, state, epsilon, use_softmax=False, temperature=1.5):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

        if use_softmax:
            # جلوگیری از overflow
            exp_q = np.exp((q_values - np.max(q_values)) / temperature)
            probs = exp_q / (np.sum(exp_q) + 1e-8)
            return np.random.choice(self.num_actions, p=probs)
        else:
            return int(np.argmax(q_values))

    def store_transition(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def sample_batch(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.sample_batch()
        states, actions, rewards, next_states = zip(*batch)

        #states = torch.tensor(states, dtype=torch.float32).to(self.device)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        #next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)


        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * max_next_q

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

