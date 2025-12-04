from collections import deque
import random
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import MSELoss

class DDPG():
    # Deep deterministic policy gradient
    # THis doesn't work
    def __init__(self, **configs):
        self.state_shape = configs.get("input_dims", 4)
        self.action_shape = configs.get("output_dims", 1)
        self.memory = deque(maxlen=configs.get("mem_len", 2048))
        self.batch_size = configs.get("batch_size", 64)
        self.gamma = configs.get("gamma", 0.99) # discount factor
        self.learning_rate = configs.get("lr", 0.01)

        self.build_actors(**configs.get("actors", {}))
        self.build_critics(**configs.get("critics", {}))

        self.loss = MSELoss()
        self.critic_optimizer = Adam(self.primary_critic.parameters())
        self.actor_optimizer = Adam(self.primary_actor.parameters())
    
    def build_network(self, input, output, hidden):
        layers = [nn.Linear(input, hidden[0]), nn.ReLU()]
        for i, _ in enumerate(hidden[:-1]):
            layers.append(nn.Linear(hidden[i], hidden[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden[-1], output))
        return nn.Sequential(*layers)
    
    def build_actors(self, **configs):
        hidden_shapes = configs.get("hidden_shapes", [256, 256])
        self.primary_actor = self.build_network(self.state_shape, self.action_shape, hidden_shapes)
        self.secondary_actor = self.build_network(self.state_shape, self.action_shape, hidden_shapes)
        self.secondary_actor.load_state_dict(self.primary_actor.state_dict())
    
    def build_critics(self, **configs):
        hidden_shapes = configs.get("hidden_shapes", [256, 256])
        self.primary_critic = self.build_network(self.state_shape + self.action_shape, 1, hidden_shapes)
        self.secondary_critic = self.build_network(self.state_shape + self.action_shape, 1, hidden_shapes)
        self.secondary_critic.load_state_dict(self.primary_critic.state_dict())

    def act(self, state, explore=0):
        action = self.primary_actor(state)
        action += torch.rand(action.shape) * explore
        return action.tolist()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        states = torch.Tensor(np.array([sample[0] for sample in batch]))
        actions = torch.Tensor(np.array([sample[1] for sample in batch]))
        rewards = torch.Tensor(np.array([sample[2] for sample in batch]))
        next_states = torch.Tensor(np.array([sample[3] for sample in batch]))
        dones = torch.Tensor(np.array([sample[4] for sample in batch]))

        target_actions = self.secondary_actor(next_states)
        future_critic_input = torch.cat((next_states, target_actions), dim=1)
        future_qs = self.secondary_critic(future_critic_input)

        target_qs = rewards + self.gamma * future_qs.squeeze().mul(1-dones)
        
        # Train the critic network
        critic_input = torch.cat((states, actions), dim=1)
        self.critic_optimizer.zero_grad()
        predict = self.primary_critic(critic_input)
        loss = self.loss(predict, target_qs)
        loss.backward()
        self.critic_optimizer.step()

        actions_for_training = self.primary_actor(states)
        actions_for_training.requires_grad_()
        critic_input_2 = torch.cat((states, actions_for_training), dim=1)
        critic_value = torch.sum(self.primary_critic(critic_input_2))

        action_gradient = torch.autograd.grad(inputs=actions_for_training, outputs=critic_value)

        actor_gradient = torch.autograd.grad(inputs=self.primary_actor.parameters(), outputs=actions_for_training, grad_outputs=action_gradient)
        print

        self.actor_optimizer.zero_grad()
        for param, grad in zip(self.primary_actor.parameters(), actor_gradient):
            param.grad = grad
        self.actor_optimizer.step()
        
        actor_primary_state = self.primary_actor.state_dict()
        actor_secondary_state = self.secondary_actor.state_dict()
        critic_primary_state = self.primary_critic.state_dict()
        critic_secondary_state = self.secondary_critic.state_dict()

        for key in actor_primary_state.keys():
            actor_secondary_state[key] += actor_primary_state[key] * 0.1
            actor_secondary_state[key] /= 1.1
        self.secondary_actor.load_state_dict(actor_secondary_state)

        for key in critic_primary_state.keys():
            critic_secondary_state[key] += critic_secondary_state[key] * 0.1
            critic_secondary_state[key] /= 1.1
        self.secondary_critic.load_state_dict(critic_secondary_state)
