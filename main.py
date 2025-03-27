import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


import gymnasium as gym
from stable_baselines3 import PPO


# Function for preprocessing each game frame. 
def preprocess(img):
    img = img[24:84, 17:77] # specific cropping
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

class ImageEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
    
    def reset(self):
        # Reset the original environment.
        s, info = self.env.reset()
        # Do nothing for the next `self.initial_no_op` steps
        for _ in range(self.initial_no_op):
            s, _, _, _, info = self.env.step(0)
        s = preprocess(s)

        # The initial observation is simply k copies of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # of size [4, 84, 84]
        return self.stacked_state, info
    
    def step(self, action):
        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info


class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim): #state_dim = number of frames = 4, action_dim = 5
        super(CNNActionValue, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 60, 60] -> [N, 16, 14, 14]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 14, 14] -> [N, 32, 6, 6]
        self.in_features = 32 * 6 * 6
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )
    

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.0004,
        epsilon=1.0,
        epsilon_min=0.1,
        gamma=0.99,
        batch_size=32,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=10000,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = CNNActionValue(state_dim[0], action_dim)
        self.target_network = CNNActionValue(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.device = torch.device('cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 6e5
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()
        return a
    
    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        
        next_q = self.target_network(s_prime).detach()
        td_target = r + (1. - terminated) * self.gamma * next_q.max(dim=1, keepdim=True).values
        self.optimizer.zero_grad()
        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
        loss.backward()
        self.optimizer.step()
        
        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }
        return result
    
    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon -= self.epsilon_decay
        return result
    


if __name__ == "__main__":

    agent_name = "PPO" # Choose between "DQN" or "PPO"
    

    if agent_name == "DQN":

        env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="rgb_array")
        env = ImageEnv(env) # New environment for DQN agent

        state_dim = (4, 60, 60)
        action_dim = env.action_space.n
        
        # Load the trained parameters of DQN agent
        DQN_agent_test = DQN(state_dim, action_dim)
        DQN_agent_test.network.load_state_dict(torch.load('true_dqn.pt'))
        DQN_agent_test.network.eval()

        ## Evaluate the DQN agent
        (s, _), done, ret = env.reset(), False, 0
        while not done:
            a = DQN_agent_test.act(s, training=False)
            s_prime, r, terminated, truncated, info = env.step(a)
            s = s_prime
            ret += r
            done = terminated or truncated
        print(f"Return for the DQN agent : {ret}")


        # Simulation of DQN agent
        env = gym.make("CarRacing-v3",domain_randomize=False, continuous=False, render_mode="human")
        env = ImageEnv(env)

        (s, _), done, ret = env.reset(), False, 0

        done = False
        env.render()

        while not done:
            a = DQN_agent_test.act(s, training=False)
            s, r, terminated, truncated, info = env.step(a)

            if 'lap_complete' in info:
                print("Track completed!")
            done = terminated or truncated
        env.close()
    


    elif agent_name == "PPO":

        env = gym.make("CarRacing-v3", domain_randomize=False, continuous=False, render_mode="rgb_array")

        # Load the trained parameters of PPO agent
        PPO_agent_test = PPO("CnnPolicy", env, verbose=1, learning_rate=1e-4, batch_size=32)
        PPO_agent_test = PPO.load("true_ppo_data")

        ## Evaluate the PPO agent
        obs, _ = env.reset()
        ret = 0
        while True:
            action, _ = PPO_agent_test.predict(obs)
            obs, r, terminated, truncated, info = env.step(np.array(action))
            ret += r
            if terminated or truncated:
                break
        print(f"Return for the PPO_agent : {ret}")


        # Simulation of PPO agent
        env = gym.make("CarRacing-v3",domain_randomize=False, continuous=False, render_mode="human")
        obs, _ = env.reset()
        env.render()

        while True:
            action, _ = PPO_agent_test.predict(obs)
            obs, r, terminated, truncated, info = env.step(np.array(action))
            if 'lap_complete' in info:
                print("Track completed!")

            if terminated or truncated:
                break
        env.close()
    
    else:
        print('Choose agent_name = "DQN" or "PPO"')