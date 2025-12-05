import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from dqn import DQN
import torch

Pendulum = "Single" # Set to "Single" or "Double"

if Pendulum == "Single":
    env = gym.make("InvertedPendulum-v5", render_mode="human", width=1280, height=720)
    action_max = 3
    input_dims = 4
elif Pendulum == "Double":
    env = gym.make("InvertedDoublePendulum-v5", render_mode="human", width=1280, height=720)
    action_max = 3
    input_dims = 4

rewards = []
maxepoch = 1000
n_actions = 11
Actor = DQN(input_dims=input_dims, output_dims=n_actions)
actionset = lambda x: [x[0][0]*(2*action_max)/n_actions - action_max]
with tqdm(range(maxepoch)) as eps:
    for epoch in eps:
        observation, info = env.reset()

        #print(f"Starting observation: {observation}")
        # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
        # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        episode_over = False
        total_reward = 0

        while not episode_over:
            # Choose an action: 0 = push cart left, 1 = push cart right
            # action = env.action_space.sample()  # Random action for now - real agents will be smarter!

            action = Actor.act(torch.Tensor(observation), explore=(1/(1+epoch)))
            #print(action)

            last_obs = observation
            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(actionset(action))

            # reward: +1 for each step the pole stays upright
            # terminated: True if pole falls too far (agent failed)
            # truncated: True if we hit the time limit (500 steps)

            total_reward += reward
            #reward -= abs(actionset(action)[0])/(10*action_max)
            episode_over = terminated or truncated
            Actor.remember(last_obs, action, reward, observation, episode_over)
            Actor.update()
        rewards.append(total_reward)
        #if total_reward > 1500:
        #    break


#print(f"Episode finished! Total reward: {total_reward}")
env.close()
plt.plot(rewards)
plt.show()

#print(Actor.memory)

