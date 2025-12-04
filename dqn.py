from collections import deque
import random
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import SmoothL1Loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DQN():
    # Deep Q network with memory and batched learning
    def __init__(self, **configs):
        self.state_shape = configs.get("input_dims", 4)
        self.action_shape = configs.get("output_dims", 2)
        self.memory = deque(maxlen=configs.get("mem_len", 2048))
        self.batch_size = configs.get("batch_size", 64)
        self.gamma = configs.get("gamma", 0.99) # discount factor
        self.learning_rate = configs.get("lr", 0.01)
        self.soft_update = configs.get("tau", 0.005)

        self.policy_net = self.build_network(**configs.get("q_net", {}))
        self.target_net = self.build_network(**configs.get("q_net", {}))
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.loss = SmoothL1Loss()
        self.optimizer = Adam(self.policy_net.parameters())

    def build_network(self, **configs):
        hidden = configs.get("hidden_shapes", [128, 128])
        input = self.state_shape
        output = self.action_shape
        layers = [nn.Linear(input, hidden[0]), nn.ReLU()]
        for i, _ in enumerate(hidden[:-1]):
            layers.append(nn.Linear(hidden[i], hidden[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden[-1], output))
        return nn.Sequential(*layers)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, explore=0):
        if random.random() < explore:
            return torch.Tensor([[random.randint(0, self.action_shape-1)]])
        return self.policy_net(state).max(0).indices.view(1, 1)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        states = torch.Tensor(np.array([sample[0] for sample in batch]))
        actions = torch.Tensor(np.array([sample[1] for sample in batch])).to(torch.int64).view(-1, 1)
        rewards = torch.Tensor(np.array([sample[2] for sample in batch]))
        next_states = torch.Tensor(np.array([sample[3] for sample in batch]))
        dones = torch.Tensor(np.array([sample[4] for sample in batch])).to(bool)
        non_final_mask = ~dones
        non_final_states = next_states[non_final_mask]
        
        state_action_values = self.policy_net(states).gather(1, actions)
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_states).max(1).values
        expected_state_Q = (next_state_values * self.gamma) + rewards
        
        self.optimizer.zero_grad()
        loss = self.loss(state_action_values, expected_state_Q.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key]*self.soft_update + target_state_dict[key] * (1-self.soft_update)
        self.target_net.load_state_dict(target_state_dict)