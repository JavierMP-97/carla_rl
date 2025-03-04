# CARLA_RL: Reinforcement Learning & Data Collection Environment for CARLA Simulator
CARLA_RL is a reinforcement learning (RL) environment designed for data collection and training autonomous driving agents within the CARLA Simulator. It provides a Gym-compatible interface, facilitating seamless interaction with RL algorithms while quickstarting autonomous driving projects.

This repository streamlines the training and evaluation process by offering built-in data collection, agent control, and simulation management tools, making it an excellent framework for both reinforcement learning and imitation learning experiments.

## Features
### Custom OpenAI Gym-Compatible Environment
- Provides a structured and flexible Reinforcement Learning (RL) training environment within the CARLA simulator.
- Implements a modular CarlaEnv class to handle simulation, agent control, and reward computation.
- Supports multiple weather conditions, dynamic object hiding, and flexible sensor configurations for realistic training scenarios.
### Autonomous Driving Data Collection
- A dedicated DataCollection module allows researchers to capture training data with minimal setup.
- Provides on-the-fly sensor data acquisition, including RGB images, semantic segmentation, depth, speed, and collision detection.
- Supports hotkeys for manual intervention:
  - **Ctrl + Q**: Exit the environment.
  - **Ctrl + W**: Reset the environment.
  - **Ctrl + E**: Reset and change the simulator map.
### Example Agents for Training
- **Autonomous Agent**: Implements customized PID-based control for longitudinal and lateral vehicle movement for data collection purposes.
  - Supports steering noise injection to enhance robustness in RL policies (noisy_control function).
- **Conditional Imitation Learning (CIL) Agent**: Demonstrates how to implement behavior cloning for imitation learning from data collection to testing.
  - Demonstrates how to fine tune a state-of-the-art backbone like MobileNetV3 for autonomous driving using TensorFlow
- **(Unfinished) Reinforcement Learning Agent**: A template for training RL-based policies using CARLA’s sensor inputs.

### Integrated Sensor Management
- Supports multiple camera types: RGB, depth, and semantic segmentation.
- Includes a collision sensor (CollisionSensor) to log crash events.
- Features speed monitoring, helping agents maintain smooth control over acceleration and braking.
- Implements agent to track center distance and agent to track direction angle difference.
### Dynamic CARLA Simulation Control
- Allows automatic map switching after a predefined number of episodes or steps.
- Enables dynamic environment variation, including randomized weather conditions and object occlusions for robust model training.
- Supports real-time rendering for visualization and debugging.

**Note**: CARLA Leaderboard and Scenario Runner must be installed alongside this library to ensure full functionality.
