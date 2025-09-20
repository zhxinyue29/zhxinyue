import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(10)  # 假设有10个离散动作
        self.observation_space = spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)  # 假设状态空间为20维

    def reset(self):
        self.state = np.random.rand(20)
        return self.state

    def step(self, action):
        reward = np.random.rand()  # 随机奖励
        done = np.random.rand() > 0.95  # 随机终止条件
        self.state = np.random.rand(20)  # 更新状态
        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass