import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import deque

# --- 1. Enhanced Plant Environment ---
class PlantSimEnv:
  
    def __init__(self):
        # State bounds: [Temp, Humidity, Moisture]
        self.low_bounds = np.array([10, 30, 10])
        self.high_bounds = np.array([40, 100, 90])
        # More granular discretization for RL state representation
        self.num_buckets = (15, 15, 15)
        # Seasonal and time-of-day
        self.season = 0  # 0=spring, 1=summer, 2=fall, 3=winter
        self.time_of_day = 0  # 0-23 hours
        self.reset()
        
    def reset(self):
     
        seasonal_temp_offset = [0, 5, 0, -5][self.season]
        seasonal_humidity_offset = [0, -10, 5, 10][self.season]
        
        self.temp = random.uniform(20, 25) + seasonal_temp_offset
        self.humidity = random.uniform(55, 65) + seasonal_humidity_offset
        self.soil_moisture = random.uniform(45, 55)
        self.plant_health = 100
        self.days_survived = 0
        self.time_of_day = 0
        self.consecutive_good_days = 0
        self.stress_level = 0  # Track cumulative stress
        
        return self._get_state()

    def _get_state(self):
     
        time_feature = np.sin(2 * np.pi * self.time_of_day / 24)
        return np.array([self.temp, self.humidity, self.soil_moisture])

    def discretize_state(self, state):
   
        state_indices = []
        for i in range(len(state)):
            clamped_val = np.clip(state[i], self.low_bounds[i], self.high_bounds[i])
            scaling = (clamped_val - self.low_bounds[i]) / (self.high_bounds[i] - self.low_bounds[i])
            new_index = int(scaling * (self.num_buckets[i] - 1))
            new_index = max(0, min(new_index, self.num_buckets[i] - 1))
            state_indices.append(new_index)
        return tuple(state_indices)

    def step(self, action):
        prev_temp, prev_humidity, prev_moisture = self.temp, self.humidity, self.soil_moisture
        
        # Action mapping:
        # 0: Do Nothing
        # 1: Water Plant (increases moisture, slight cooling from evaporation)
        # 2: Run Ventilation (decreases humidity and temp)
        # 3: Run Heater (increases temp, decreases humidity slightly)
        # 4: Run Humidifier (increases humidity, slight temp increase)
        action_effects = {
            0: {"moisture": 0, "temp": 0, "humidity": 0, "cost": 0},
            1: {"moisture": 12, "temp": -0.5, "humidity": 3, "cost": 0.3},
            2: {"moisture": -1, "temp": -1.5, "humidity": -15, "cost": 0.8},
            3: {"moisture": -2, "temp": 3, "humidity": -3, "cost": 1.2},
            4: {"moisture": 0, "temp": 0.5, "humidity": 12, "cost": 0.6},
        }
        
        if action in action_effects:
            effects = action_effects[action]
            self.soil_moisture += effects["moisture"]
            self.temp += effects["temp"]
            self.humidity += effects["humidity"]
            action_cost = effects["cost"]
        else:
            action_cost = 0

        # Time-of-day effects
        self.time_of_day = (self.time_of_day + 1) % 24
        if 6 <= self.time_of_day <= 18:  # Daytime: temperature tends to rise
            temp_drift = random.uniform(0.2, 0.8)
            humidity_drift = random.uniform(-1, 1)
        else:  # Night: cooler with slightly different humidity behavior
            temp_drift = random.uniform(-0.5, 0.1)
            humidity_drift = random.uniform(1, 3)

        self.temp += temp_drift + random.uniform(-0.3, 0.3)
        self.humidity += humidity_drift + random.uniform(-2, 4)
        self.soil_moisture -= random.uniform(0.5, 2.5)  # Natural evapotranspiration

        # Seasonal progression (weekly chance to shift)
        if self.days_survived % 7 == 0:
            self.season = (self.season + random.choice([0, 1])) % 4

        # Physical constraints
        self.temp = np.clip(self.temp, -5, 55)
        self.humidity = np.clip(self.humidity, 0, 100)
        self.soil_moisture = np.clip(self.soil_moisture, 0, 100)

        self.days_survived += 1

        # Reward + termination
        reward = self._calculate_reward(action, action_cost, prev_temp, prev_humidity, prev_moisture)
        done = self._check_termination()
        
        return self._get_state(), reward, done

    def _calculate_reward(self, action, action_cost, prev_temp, prev_humidity, prev_moisture):
        reward = 0
        
        # Tighter ideal ranges encourage precision
        ideal_temp_range = (22, 26)
        ideal_humidity_range = (60, 75)
        ideal_moisture_range = (50, 70)
        
        # Base rewards for being in ideal conditions (smooth via gaussian)
        temp_reward = self._gaussian_reward(self.temp, ideal_temp_range[0], ideal_temp_range[1], max_reward=3)
        humidity_reward = self._gaussian_reward(self.humidity, ideal_humidity_range[0], ideal_humidity_range[1], max_reward=3)
        moisture_reward = self._gaussian_reward(self.soil_moisture, ideal_moisture_range[0], ideal_moisture_range[1], max_reward=3)
        reward += temp_reward + humidity_reward + moisture_reward
        
        # Stability bonus for avoiding rapid temp changes
        stability_bonus = 0
        temp_change = abs(self.temp - prev_temp)
        if temp_change < 1.0:
            stability_bonus += 1
        
        # Penalties for extreme/worsening conditions (increase stress)
        if self.temp < 15 or self.temp > 32:
            reward -= 5
            self.stress_level += 2
        if self.humidity > 85 or self.humidity < 40:
            reward -= 3
            self.stress_level += 1
        if self.soil_moisture < 25:
            reward -= 8
            self.stress_level += 3
        elif self.soil_moisture > 85:
            reward -= 4
            self.stress_level += 1

        # Efficiency penalty to discourage excessive actions
        reward -= action_cost
        
        # Consistency bonus for sustained ideal conditions (caps to avoid runaway)
        if all([
            ideal_temp_range[0] <= self.temp <= ideal_temp_range[1],
            ideal_humidity_range[0] <= self.humidity <= ideal_humidity_range[1],
            ideal_moisture_range[0] <= self.soil_moisture <= ideal_moisture_range[1]
        ]):
            self.consecutive_good_days += 1
            reward += min(self.consecutive_good_days * 0.5, 5)
        else:
            self.consecutive_good_days = 0
        
        # Plant health update: stress reduces health; good rewards slowly recover it
        if self.stress_level > 10:
            self.plant_health -= 2
            self.stress_level = max(0, self.stress_level - 1)
        elif reward > 5:
            self.plant_health = min(100, self.plant_health + 0.5)
            
        return reward + stability_bonus

    def _gaussian_reward(self, value, ideal_min, ideal_max, max_reward=3):
        """Gaussian-shaped reward for smooth gradients away from ideal range."""
        ideal_center = (ideal_min + ideal_max) / 2
        ideal_range = ideal_max - ideal_min
        if ideal_min <= value <= ideal_max:
            return max_reward
        distance = min(abs(value - ideal_min), abs(value - ideal_max))
        return max_reward * np.exp(-(distance / ideal_range) ** 2)

    def _check_termination(self):
        """Episode ends if plant dies, max days reached, or stress too high."""
        if self.plant_health <= 0:
            return True
        if self.days_survived >= 150:  # Longer episodes to allow more learning
            return True
        if self.stress_level > 50:  # Excessive cumulative stress
            return True
        return False

# --- 2. Enhanced Q-Learning Agent ---
class AdvancedQLearningAgent:
  
    def __init__(self, state_space_shape, action_space_size, alpha=0.1, gamma=0.95, 
                 epsilon=1.0, use_adaptive_lr=True):
        self.q_table = np.zeros(state_space_shape + (action_space_size,))
        self.visit_counts = np.zeros(state_space_shape + (action_space_size,))
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.use_adaptive_lr = use_adaptive_lr
        
        self.episode_rewards = []
        self.episode_lengths = []
        
    def choose_action(self, state, use_boltzmann=False, temperature=1.0):
        """Boltzmann exploration is used in later training; otherwise epsilon-greedy."""
        if use_boltzmann and self.epsilon < 0.3:
            q_values = self.q_table[state]
            exp_values = np.exp(q_values / temperature)
            probabilities = exp_values / np.sum(exp_values)
            return np.random.choice(len(probabilities), p=probabilities)
        else:
            # Epsilon-greedy
            if random.uniform(0, 1) < self.epsilon:
                return random.randint(0, self.q_table.shape[-1] - 1)
            else:
                return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """Adaptive learning rate using visit counts; standard Q-learning update."""
        self.visit_counts[state][action] += 1
        
        if self.use_adaptive_lr:
            adaptive_alpha = self.alpha / (1 + 0.001 * self.visit_counts[state][action])
        else:
            adaptive_alpha = self.alpha
            
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + adaptive_alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

        # Decay exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_policy(self, state):
        """Return greedy action for given state (policy extraction)."""
        return np.argmax(self.q_table[state])

# --- 3. Enhanced Training Loop with Monitoring ---
def train_agent(num_episodes=25000, save_plots=True):
    env = PlantSimEnv()
    action_space_size = 5  # [Do Nothing, Water, Ventilation, Heat, Humidify]
    state_space_shape = env.num_buckets

    agent = AdvancedQLearningAgent(state_space_shape, action_space_size)

    print("Starting enhanced RL agent training...")
    print(f"State space: {state_space_shape}")
    print(f"Action space: {action_space_size}")

    reward_window = deque(maxlen=100)
    best_avg_reward = -float('inf')

    for episode in range(num_episodes):
        continuous_state = env.reset()
        state = env.discretize_state(continuous_state)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Use Boltzmann exploration in later training
            action = agent.choose_action(state, use_boltzmann=(episode > num_episodes//2))
            next_continuous_state, reward, done = env.step(action)
            next_state = env.discretize_state(next_continuous_state)

            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward
            steps += 1

        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        reward_window.append(total_reward)
        
        if len(reward_window) == 100:
            avg_reward = np.mean(reward_window)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(reward_window) if reward_window else total_reward
            print(f"Episode: {episode + 1}/{num_episodes}")
            print(f"  Total Reward: {total_reward:.1f}")
            print(f"  Avg Reward (100): {avg_reward:.1f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Plant Health: {env.plant_health:.1f}")
            print(f"  Days Survived: {env.days_survived}")

    print("Training finished!")
    print(f"Best average reward (100 episodes): {best_avg_reward:.1f}")

    model_data = {
        'q_table': agent.q_table,
        'state_space_shape': state_space_shape,
        'action_space_size': action_space_size,
        'low_bounds': env.low_bounds,
        'high_bounds': env.high_bounds,
        'num_buckets': env.num_buckets
    }
    
    with open('advanced_q_table_greenhouse.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Enhanced model saved to 'advanced_q_table_greenhouse.pkl'")
    
    if save_plots:
        plot_training_results(agent, num_episodes)
    
    return agent, env

def plot_training_results(agent, num_episodes):
    """Generate training performance plots (reward curve, episode length, distribution, Q-heatmap)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward over time with moving average
    ax1.plot(agent.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(agent.episode_rewards) > 100:
        moving_avg = []
        for i in range(99, len(agent.episode_rewards)):
            moving_avg.append(np.mean(agent.episode_rewards[i-99:i+1]))
        ax1.plot(range(99, len(agent.episode_rewards)), moving_avg, 
                 color='red', linewidth=2, label='100-Episode Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Episode length over time
    ax2.plot(agent.episode_lengths, alpha=0.6, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Duration')
    ax2.grid(True)
    
    # Reward distribution
    ax3.hist(agent.episode_rewards, bins=50, alpha=0.7, color='purple')
    ax3.set_xlabel('Total Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Reward Distribution')
    ax3.grid(True)
    
    # Q-value heatmap for a slice of the Q-table (mid moisture level/action slice)
    q_slice = np.max(agent.q_table, axis=2)  # Max Q-value across moisture states
    im = ax4.imshow(q_slice[:, :, q_slice.shape[2]//2], cmap='viridis', aspect='auto')
    ax4.set_xlabel('Humidity State')
    ax4.set_ylabel('Temperature State')
    ax4.set_title('Learned Q-Values (Mid Moisture Level)')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('greenhouse_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- 4. Testing and Evaluation ---
def test_agent(agent, env, num_test_episodes=10):
    """Test the trained agent's performance using greedy policy (pure exploitation)."""
    print("\n--- Testing Trained Agent ---")
    
    test_rewards = []
    test_lengths = []
    
    for episode in range(num_test_episodes):
        continuous_state = env.reset()
        state = env.discretize_state(continuous_state)
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 200:  # Safety cap to prevent infinite loops
            action = agent.get_policy(state)
            next_continuous_state, reward, done = env.step(action)
            next_state = env.discretize_state(next_continuous_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        
        print(f"Test Episode {episode + 1}: Reward = {total_reward:.1f}, "
              f"Length = {steps}, Final Health = {env.plant_health:.1f}")
    
    print(f"\nTest Results:")
    print(f"Average Reward: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print(f"Average Length: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")

if __name__ == '__main__':
    agent, env = train_agent(num_episodes=25000)
    
    # Test the trained agent
    test_agent(agent, env)
    
    print("\nTraining and testing complete!")
    print("Files saved:")
    print("- advanced_q_table_greenhouse.pkl (trained model)")
    print("- greenhouse_training_results.png (training plots)")
