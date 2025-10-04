import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Config:
    
    NUM_EPISODES = 25000
    MAX_STEPS_PER_EPISODE = 150

    # DQN Agent parameters
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    GAMMA = 0.99
    LR = 1e-5                
    TAU = 1e-4               
    UPDATE_EVERY = 8

    # Exploration parameters
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.9999


class PlantSimEnv:
   
    def __init__(self):
     
        self.low_bounds = np.array([10, 30, 10])  # [Temp, Humidity, Moisture]
        self.high_bounds = np.array([40, 100, 90])
    
        self.state_space_dim = len(self.low_bounds)

        # Initialize seasonal and time-of-day variation before reset
        self.season = 0  # 0=spring, 1=summer, 2=fall, 3=winter
        self.time_of_day = 0  # 0-23 hours
        self.reset()

    def reset(self):
        # Resets the environment with seasonal variation.
        seasonal_temp_offset = [0, 5, 0, -5][self.season]
        seasonal_humidity_offset = [0, -10, 5, 10][self.season]

        self.temp = random.uniform(20, 25) + seasonal_temp_offset
        self.humidity = random.uniform(55, 65) + seasonal_humidity_offset
        self.soil_moisture = random.uniform(45, 55)
        self.plant_health = 100
        self.days_survived = 0
        self.time_of_day = 0
        self.consecutive_good_days = 0
        self.stress_level = 0
        return self._get_state()

    def _get_state(self):
       
        state = np.array([self.temp, self.humidity, self.soil_moisture])
        # Normalize state to be between -1 and 1 for better network performance
        normalized_state = 2 * (state - self.low_bounds) / (self.high_bounds - self.low_bounds) - 1
        return normalized_state

   
    def step(self, action):
        
        prev_temp, prev_humidity, prev_moisture = self.temp, self.humidity, self.soil_moisture

        
        action_effects = {
            0: {"moisture": 0, "temp": 0, "humidity": 0, "cost": 0},
            1: {"moisture": 12, "temp": -0.5, "humidity": 3, "cost": 0.3},  # Watering
            2: {"moisture": -1, "temp": -1.5, "humidity": -15, "cost": 0.8},  # Ventilation
            3: {"moisture": -2, "temp": 3, "humidity": -3, "cost": 1.2},  # Heating
            4: {"moisture": 0, "temp": 0.5, "humidity": 12, "cost": 0.6},  # Humidification
        }

        if action in action_effects:
            effects = action_effects[action]
            self.soil_moisture += effects["moisture"]
            self.temp += effects["temp"]
            self.humidity += effects["humidity"]
            action_cost = effects["cost"]
        else:
            action_cost = 0

       
        self.time_of_day = (self.time_of_day + 1) % 24
        if 6 <= self.time_of_day <= 18:
            temp_drift = random.uniform(0.2, 0.8)
            humidity_drift = random.uniform(-1, 1)
        else:
            temp_drift = random.uniform(-0.5, 0.1)
            humidity_drift = random.uniform(1, 3)

        self.temp += temp_drift + random.uniform(-0.3, 0.3)
        self.humidity += humidity_drift + random.uniform(-2, 4)
        self.soil_moisture -= random.uniform(0.5, 2.5)

        if self.days_survived % 7 == 0:
            self.season = (self.season + random.choice([0, 1])) % 4

        self.temp = np.clip(self.temp, -5, 55)
        self.humidity = np.clip(self.humidity, 0, 100)
        self.soil_moisture = np.clip(self.soil_moisture, 0, 100)
        self.days_survived += 1

        reward = self._calculate_reward(action, action_cost, prev_temp, prev_humidity, prev_moisture)
        done = self._check_termination()

        return self._get_state(), reward, done

    def _calculate_reward(self, action, action_cost, prev_temp, prev_humidity, prev_moisture):
        reward = 0
        ideal_temp_range = (22, 26)
        ideal_humidity_range = (60, 75)
        ideal_moisture_range = (50, 70)

        reward += self._gaussian_reward(self.temp, ideal_temp_range[0], ideal_temp_range[1])
        reward += self._gaussian_reward(self.humidity, ideal_humidity_range[0], ideal_humidity_range[1])
        reward += self._gaussian_reward(self.soil_moisture, ideal_moisture_range[0], ideal_moisture_range[1])

        stability_bonus = 1 if abs(self.temp - prev_temp) < 1.0 else 0

        if self.temp < 15 or self.temp > 32:
            reward -= 5; self.stress_level += 2
        if self.humidity > 85 or self.humidity < 40:
            reward -= 3; self.stress_level += 1
        if self.soil_moisture < 25:
            reward -= 8; self.stress_level += 3
        elif self.soil_moisture > 85:
            reward -= 4; self.stress_level += 1

        reward -= action_cost

        if all([
            ideal_temp_range[0] <= self.temp <= ideal_temp_range[1],
            ideal_humidity_range[0] <= self.humidity <= ideal_humidity_range[1],
            ideal_moisture_range[0] <= self.soil_moisture <= ideal_moisture_range[1]
        ]):
            self.consecutive_good_days += 1
            reward += min(self.consecutive_good_days * 0.5, 5)
        else:
            self.consecutive_good_days = 0

        if self.stress_level > 10:
            self.plant_health -= 2
            self.stress_level = max(0, self.stress_level - 1)
        elif reward > 5:
            self.plant_health = min(100, self.plant_health + 0.5)

        return reward + stability_bonus

    def _gaussian_reward(self, value, ideal_min, ideal_max, max_reward=3):
        ideal_center = (ideal_min + ideal_max) / 2
        ideal_range = ideal_max - ideal_min
        if ideal_min <= value <= ideal_max:
            return max_reward
        else:
            distance = min(abs(value - ideal_min), abs(value - ideal_max))
            return max_reward * np.exp(-(distance / ideal_range) ** 2)

    def _check_termination(self):
        if self.plant_health <= 0: return True
        if self.days_survived >= Config.MAX_STEPS_PER_EPISODE: return True
        if self.stress_level > 50: return True
        return False

# --- 3. DQN Model ---
class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
       
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    
    def __init__(self, buffer_size, batch_size, seed=0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# --- 5. DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.epsilon = config.EPS_START

        # Q-Network
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)

        # Initialize target network with policy network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Set target network to evaluation mode

        # Replay memory
        self.memory = ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.policy_net.eval() # Set network to evaluation mode for inference
            with torch.no_grad():
                action_values = self.policy_net(state)
            self.policy_net.train() # Set it back to train mode
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences

        # --- Double DQN Modification ---
        # 1. Get the best action for the next_state from the *policy* network.
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)

        # 2. Get the Q-value for that best_action from the *target* network.
        q_targets_next = self.target_net(next_states).gather(1, best_actions)
        # --- End of Modification ---

        # Compute Q targets for current states
        q_targets = rewards + (self.config.GAMMA * q_targets_next * (1 - dones))

        # Get expected Q values from policy model
        q_expected = self.policy_net(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network using a soft update
        self.soft_update_target_network()

        # Update epsilon
        self.epsilon = max(self.config.EPS_END, self.config.EPS_DECAY * self.epsilon)

    def soft_update_target_network(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.config.TAU * policy_param.data + (1.0 - self.config.TAU) * target_param.data)

# --- 6. Enhanced Training Loop ---
def train_dqn_agent(save_plots=True):
    config = Config()
    env = PlantSimEnv()
    action_space_size = 5
    state_space_size = env.state_space_dim

    agent = DQNAgent(state_space_size, action_space_size, config)

    print("Starting DQN agent training...")
    print(f"State space size: {state_space_size}, Action space size: {action_space_size}")

    episode_rewards = []
    episode_lengths = []
    reward_window = deque(maxlen=100)
    best_avg_reward = -float('inf')

    for episode in range(1, config.NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0

        for t in range(config.MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.step(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(t + 1)
        reward_window.append(total_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(reward_window)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.policy_net.state_dict(), 'best_dqn_greenhouse.pth')

            print(f"\rEpisode {episode}\tAvg Reward: {avg_reward:.2f}\tEpsilon: {agent.epsilon:.4f}", end="")

        if episode % 1000 == 0:
            print(f"\rEpisode {episode}\tAvg Reward: {np.mean(reward_window):.2f}\tEpsilon: {agent.epsilon:.4f}")

    print("\nTraining finished!")
    print(f"Best average reward (100 episodes): {best_avg_reward:.2f}")

    if save_plots:
        plot_training_results(episode_rewards, episode_lengths)

    return agent, env

def plot_training_results(rewards, lengths):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Reward over time with moving average
    ax1.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    moving_avg = [np.mean(rewards[i-99:i+1]) for i in range(99, len(rewards))]
    ax1.plot(range(99, len(rewards)), moving_avg, color='red', linewidth=2, label='100-Episode Average')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward'); ax1.set_title('Training Progress')
    ax1.legend(); ax1.grid(True)

    # Plot 2: Reward distribution
    ax2.hist(rewards, bins=50, alpha=0.7, color='purple')
    ax2.set_xlabel('Total Reward'); ax2.set_ylabel('Frequency'); ax2.set_title('Reward Distribution')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('greenhouse_dqn_training_results.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    train_dqn_agent()
    print("\nTraining complete! Model saved to 'best_dqn_greenhouse.pth'")
