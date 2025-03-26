# Solving_Car_Racing_v3

*Authors*: Adrien Fu, Raphaël Faure  
*Date*: March 2025

## Repository Structure

```
.
├── CarRacing_v3_Fu_Faure.ipynb     # Main Notebook illustrating our work
├── main.py                         # DQN training script
├── true_dqn.pt                     # Trained DQN model
├── true_output_PPO.txt             # PPO training logs
├── true_ppo_data.zip               # PPO training data and model
├── Report_RL.pdf                   # Full project report
├── Poster RL.pdf                   # Summary poster
├── README.md                       # Project documentation
├── images/                         # Folder containing visuals
└── .DS_Store                       # System file (can be ignored)
```



## Overview

This project uses Deep Reinforcement Learning to train agents to solve the CarRacing-v3 environment from Gymnasium. We compare two algorithms:

- **DQN (Deep Q-Network)** implemented from scratch
- **PPO (Proximal Policy Optimization)** using Stable-Baselines3

The goal is to teach agents to drive efficiently with partial observability.

## Environment

- **CarRacing-v3 (Discrete)**
- **Observation**: 96x96 RGB frames
- **Action space**: 5 discrete actions (nothing, left, right, gas, brake)
- **Rewards**: +1000/N per tile visited, -0.1 per frame, -100 if off track
  
The run example given in the gif is rendered in full-scale mode i.e., that is what a human sees playing this game.<br>

<p align='center'>
<img src='./images/run.gif'>
</p>


## Methods
Here are our two agents :
### DQN

<p align='center'>
<img src='./images/dqn architecture.png'>
</p>

- Custom implementation with experience replay and target network
- Architecture: CNN → 2 FC layers
- 600,000 steps (~12h), final average reward: **605**
  

### PPO

<p align='center'>
<img src='./images/ppo architecture.png'>
</p>

- Stable-Baselines3 with CNN policy
- 650,000 steps (~3h), final average reward: **733**
- Faster convergence, but temporary policy collapse observed

### Results

<p align="center">
  <img src="./images/dqn agent.png" alt="DQN Agent" width="45%"/>
  <img src="./images/ppo agent.png" alt="PPO Agent" width="45%"/>
</p>


| Agent | Steps | Time   | Final Avg Return | Stability       |
|-------|-------|--------|------------------|-----------------|
| DQN   | 600k  | 12 hrs | 605              | High variance   |
| PPO   | 650k  | 3 hrs  | 733              | More stable, some collapse |

## References

[1] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, M. Riedmiller (2013). *Playing Atari with Deep Reinforcement Learning*. DeepMind Technologies. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

[2] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov (2017). *Proximal Policy Optimization Algorithms*. OpenAI. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[3] A. Fu, R. Faure (2025). *Solving Car Racing v3*. GitHub Repository. [GitHub Link](https://github.com/Adri4000/Solving_Car_Racing_v3)

[4] Farama-Foundation (2023). *Gymnasium: A toolkit for developing and comparing reinforcement learning algorithms*. [GitHub](https://github.com/Farama-Foundation/Gymnasium)

[5] A. Zai, B. Brown (2020). *Deep Reinforcement Learning in Action*. Manning, Chapter 3. [Online Resource](https://livebook.manning.com/concept/deep-learning/q-network)

[6] A. Raffin et al. (2021). *Stable-Baselines3: A set of reliable implementations of reinforcement learning algorithms*. [GitHub](https://github.com/DLR-RM/stable-baselines3)

[7] S. Moalla, A. Miele, D. Pyatko, R. Pascanu, C. Gulcehre (2024). *No Representation, No Trust: Connecting Representation, Collapse, and Trust Issues in PPO*. CLAIRE EPFL, Google DeepMind. [arXiv:2405.00662](https://arxiv.org/pdf/2405.00662)


---







