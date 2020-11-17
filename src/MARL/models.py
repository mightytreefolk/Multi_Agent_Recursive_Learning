import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import gym

from scipy.interpolate import griddata
from random import randrange


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr = self.mem_cntr + 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch].to(self.Q_eval.device))

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min


class FreeEnergyBarrier(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, m, n):

        self.grid = self.makeGrid(m, n)
        self.m = m
        self.n = n
        self.stateSpace = [[x for x in range(self.m)],
                           [y for y in range(self.n)]]
        """
        0 -> X 
        1 -> -X 
        2 -> Y
        3 -> -Y
        """
        self.actionSpace = {0: [1, 0], 1: [-1, 0], 2: [0, 1], 3: [0, -1]}
        self.possibleActions = [0, 1, 2, 3]
        self.agentPosition = np.array([randrange(self.m), randrange(self.n)])
        self.viewer = None

    def gridFunction(self, x, y):
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

    def isTerminalState(self):
        if self.agentPosition[0] == 152 and self.agentPosition[1] == 68:
            return True
        else:
            return False

    def makeGrid(self, x, y):
        m = np.linspace(0, 1, x)
        n = np.linspace(0, 1, y)
        grid_x, grid_y = np.meshgrid(m, n)
        points = np.random.rand(x * y * 10, 2)
        values = self.gridFunction(points[:, 0], points[:, 1])
        grid = pd.DataFrame(griddata(points, values, (grid_x.T, grid_y.T), method='cubic'))
        return grid

    def offGridMove(self, newState, oldState):
        # if we move into a row not in the grid
        if newState[0] not in self.stateSpace[0] or newState[1] not in self.stateSpace[1]:
            return True
        elif newState[0] < 0 or newState[1] < 0:
            return True
        elif newState[0] >= 200 or newState[1] >= 200:
            return True
        else:
            return False

    def agentCurrentPosition(self):
        x, y = self.agentPosition[0], self.agentPosition[1]
        return np.array([x, y])

    def setState(self, state):
        self.agentPosition = state

    def step(self, action):
        resultingState = []
        for i, j in zip(self.agentPosition, self.actionSpace[action]):
            resultingState.append(i+j)
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            if self.grid[resultingState[0]][resultingState[1]] < 0:
                reward = 1
            elif self.isTerminalState():
                reward = 100
            elif self.grid[resultingState[0]][resultingState[1]] > 0:
                reward = -1
            else:
                reward = -1
            return resultingState, reward, self.isTerminalState(), {}
        else:
            self.setState(self.agentPosition)
            if self.grid[self.agentPosition[0]][self.agentPosition[1]] < 0:
                reward = 0
            elif self.isTerminalState():
                reward = 100
            elif self.grid[self.agentPosition[0]][self.agentPosition[1]] > 0:
                reward = -1
            else:
                reward = 0
            return self.agentPosition, reward, self.isTerminalState(), {}

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

    def reset(self):
        self.grid = self.makeGrid(self.m, self.n)
        self.agentPosition = np.array([randrange(self.m), randrange(self.n)])
        return self.agentPosition

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

