# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#


# Abdessamad Badaoui 20011228-T118
# Nasr Allah Aghelias 20010616-T318

# Load packages
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
from Classes import Model, Experience, ExperienceReplayBuffer

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 600                             # Number of episodes
discount_factor = 0.98                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = RandomAgent(n_actions)


input_size = dim_state
hidden_size1 = 32
hidden_size2 = 32
output_size = n_actions

# Instantiate the neural network
main_network = Model(input_size, hidden_size1, hidden_size2, output_size).double()
target_network = Model(input_size, hidden_size1, hidden_size2, output_size).double()

# Example of using the buffer
buffer_capacity = 20000
experience_buffer = ExperienceReplayBuffer(maximum_length=buffer_capacity)

for _ in range(buffer_capacity):
    state = np.random.rand(dim_state) * 10
    action = np.random.randint(0, n_actions, size=(1, ))
    reward = np.array([-200]) * (1 + np.random.rand() - 0.5)
    next_state = np.random.rand(dim_state) * 10
    done = np.random.randint(0, 2, size=(1, ))

    exp = Experience(state, action, reward, next_state, done)

    experience_buffer.append(exp)


def Q_value(network, state):
    input_value = state
    network.eval()
    with torch.no_grad():
        output = network(input_value)
    network.train()
    return output


def epsilon_greedy(epsilon, network, state):
    p = np.random.random()
    if p < epsilon:
        return torch.randint(0, n_actions, size=(1, )).item()
    else:
        values = Q_value(network, state)
        return torch.argmax(values).item()

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

learning_rate = 5e-4
optimizer = optim.Adam(main_network.parameters(), lr=learning_rate)

batch_size = 32
C = buffer_capacity // batch_size
clipping_value = 1

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    epsilon = max(0.05, 0.99 - (0.94 * i / N_episodes))
    while not done:
        action = epsilon_greedy(epsilon, main_network, torch.tensor(state, dtype=torch.float64))

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        exp = Experience(state, np.array([action]), np.array([reward]), next_state, np.array([int(terminated)]))

        experience_buffer.append(exp)

        states, actions, rewards, next_states, dones = experience_buffer.sample_batch(batch_size)

        states = torch.tensor(np.array(states), dtype=torch.float64)
        actions = torch.tensor(np.array(actions))
        rewards = torch.tensor(np.array(rewards))
        next_states = torch.tensor(np.array(next_states), dtype=torch.float64)
        dones = torch.tensor(np.array(dones))

        q_values_target = target_network(next_states).detach()

        y_values = rewards + discount_factor * torch.max(q_values_target, dim=1)[0].view(-1, 1) * (1 - dones)

        q_values = main_network(states).gather(1, actions)

        loss = F.mse_loss(q_values, y_values)

        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(main_network.parameters(), clipping_value)
        optimizer.step() 

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1
        if t%C:
            for target_param, main_param in zip(target_network.parameters(), main_network.parameters()):
                target_param.data.copy_(main_param.data)

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))


torch.save(main_network, 'neural-network-1.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
