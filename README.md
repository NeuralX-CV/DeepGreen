# 🌱 Greenhouse IoT DQN Control System

A Deep Q-Network (DQN) based intelligent control system for automated greenhouse environment management. This system uses reinforcement learning to optimize plant growth conditions by controlling temperature, humidity, and soil moisture through various actuators.

## 🚀 Features

- **Deep Reinforcement Learning**: Uses DQN with Double DQN implementation for stable learning
- **Multi-Parameter Control**: Manages temperature, humidity, and soil moisture simultaneously
- **Realistic Environment Simulation**: Includes seasonal variations, time-of-day effects, and plant stress modeling
- **IoT Ready**: Designed for integration with real greenhouse sensor and actuator systems
- **Comprehensive Monitoring**: Tracks plant health, stress levels, and environmental stability

## 📊 Performance

The latest model achieves:
- **Peak Performance**: ~1522 average reward over 100 episodes
- **Stable Convergence**: Consistent performance in 1100-1500 range
- **Training Stability**: No catastrophic forgetting or policy degradation
- **Exploration Strategy**: Proper epsilon decay from exploration to exploitation

![Training Progress](training_progress_example.png)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/greenhouse-dqn-control.git
cd greenhouse-dqn-control

# Create virtual environment
python -m venv greenhouse_env
source greenhouse_env/bin/activate  # On Windows: greenhouse_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0
```

## 🎮 Quick Start

### Training a New Model

```python
from greenhouse_dqn import train_dqn_agent

# Train with default hyperparameters
agent, env = train_dqn_agent(save_plots=True)
```

### Using Pre-trained Model

```python
import torch
from greenhouse_dqn import QNetwork, PlantSimEnv, DQNAgent, Config

# Load environment and agent
env = PlantSimEnv()
config = Config()
agent = DQNAgent(state_size=3, action_size=5, config=config)

# Load pre-trained weights
agent.policy_net.load_state_dict(torch.load('best_dqn_greenhouse.pth'))
agent.policy_net.eval()

# Run inference
state = env.reset()
action = agent.choose_action(state)
```

## 🏗️ System Architecture

### State Space (3D Continuous)
- **Temperature**: 10-40°C (normalized to [-1, 1])
- **Humidity**: 30-100% (normalized to [-1, 1])  
- **Soil Moisture**: 10-90% (normalized to [-1, 1])

### Action Space (5 Discrete Actions)
- **0**: No action
- **1**: Watering (+12% moisture, -0.5°C temp, +3% humidity)
- **2**: Ventilation (-15% humidity, -1.5°C temp, -1% moisture)
- **3**: Heating (+3°C temp, -3% humidity, -2% moisture)
- **4**: Humidification (+12% humidity, +0.5°C temp)

### Reward Function
- **Gaussian rewards** for optimal ranges:
  - Temperature: 22-26°C
  - Humidity: 60-75%
  - Soil Moisture: 50-70%
- **Penalties** for extreme conditions and excessive actions
- **Bonuses** for consecutive days in optimal conditions
- **Stability rewards** for smooth environmental transitions

## 🔧 Configuration

Key hyperparameters in `Config` class:

```python
class Config:
    NUM_EPISODES = 25000
    MAX_STEPS_PER_EPISODE = 150
    
    # DQN Parameters
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    GAMMA = 0.99
    LR = 5e-5              # Learning rate
    TAU = 5e-4             # Soft update rate
    UPDATE_EVERY = 4       # Network update frequency
    
    # Exploration
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.9995
```

## 📈 Training Results

### Latest Model Performance:
- **Episodes 1-3000**: Learning phase with exploration
- **Episodes 4000-11000**: Rapid improvement to peak performance
- **Episodes 11000+**: Stable policy maintenance
- **Final Performance**: Consistent 1100-1500 reward range

### Key Metrics:
- **Convergence Time**: ~10,000 episodes
- **Peak Average Reward**: 1522 (100-episode window)
- **Training Stability**: No catastrophic forgetting
- **Policy Robustness**: Maintained performance over 15,000+ episodes

## 🏭 IoT Integration

### Sensor Integration
The system expects normalized sensor readings:
```python
# Example sensor reading integration
temperature = read_temperature_sensor()  # °C
humidity = read_humidity_sensor()        # %
moisture = read_soil_moisture_sensor()   # %

# Normalize for model input
state = normalize_sensor_data([temperature, humidity, moisture])
action = agent.choose_action(state)
```

### Actuator Control
Actions map to physical actuator commands:
```python
def execute_action(action):
    if action == 1:      # Watering
        activate_irrigation_pump(duration=30)
    elif action == 2:    # Ventilation
        open_ventilation_fans()
    elif action == 3:    # Heating
        activate_heater()
    elif action == 4:    # Humidification
        activate_humidifier()
    # action == 0: No action needed
```

## 🔒 Safety Features

### Built-in Safeguards
- **Hard limits** on environmental parameters
- **Stress level monitoring** prevents plant damage
- **Action cost penalties** prevent excessive resource usage
- **Consecutive good days bonus** encourages stability

### Deployment Recommendations
1. **Gradual rollout** with human oversight
2. **Backup manual controls** for emergency situations
3. **Real-time monitoring** of model decisions
4. **Regular model retraining** with real greenhouse data


## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Model Validation
```bash
python examples/test_model.py --model best_dqn_greenhouse.pth --episodes 100
```

### Stress Testing
```bash
python examples/stress_test.py --extreme-conditions
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Test with multiple random seeds

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Research & References

- [Deep Q-Network (DQN) Paper](https://www.nature.com/articles/nature14236)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- Greenhouse automation research and IoT best practices

## 🆘 Troubleshooting

### Common Issues

**Training Instability**
- Reduce learning rate (`LR = 1e-5`)
- Increase target network update frequency (`TAU = 1e-4`)
- Add gradient clipping

**Poor Convergence**
- Increase network size or add layers
- Adjust reward function parameters
- Increase replay buffer size

**IoT Integration Issues**
- Ensure proper sensor calibration
- Check actuator response times
- Validate state normalization

## 📞 Support

- Create an issue for bugs or feature requests
- Join our discussions for questions and ideas
- Check the [Wiki](wiki) for detailed documentation

## 🏆 Acknowledgments

- Plant biology expertise from agricultural research community
- IoT sensor integration patterns from greenhouse automation projects
- Deep reinforcement learning implementation best practices

---

**⚠️ Important**: This system is designed for research and development purposes. Always maintain manual oversight when deploying in production greenhouse environments.
