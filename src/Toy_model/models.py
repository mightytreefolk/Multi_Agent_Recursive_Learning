import numpy as np
from random import randrange
import gym
from scipy.interpolate import griddata
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from gym import error, spaces, utils
from gym.utils import seeding


class FreeEnergyBarrier(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, m, n):

        self.grid = self.makeGrid(m, n)
        self.m = m
        self.n = n
        self.stateSpace = [[x for x in range(self.m)],
                           [y for y in range(self.n)]]
        self.actionSpace = {"X": [1, 0], "-X": [-1, 0], "Y": [0, 1], "-Y": [0, -1]}
        self.possibleActions = ['X', '-X', 'Y', '-Y']
        self.agentPosition = np.array([randrange(self.m), randrange(self.n)])
        self.viewer = None

    def gridFunction(self, x, y):
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

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
        elif newState[0] >= self.m or newState[1] >= self.n:
            return True
        else:
            return False

    def agentCurrentPosition(self):
        x, y = self.agentPosition[0], self.agentPosition[1]
        return np.array([x, y])

    def setState(self, state):
        self.agentPosition = state

    def step(self, action: str):
        resultingState = []
        for i, j in zip(self.agentPosition, self.actionSpace[action]):
            resultingState.append(i+j)
        # resultingState = np.array(resultingState)
        if not self.offGridMove(resultingState, self.agentPosition):
            reward = self.grid[resultingState[0]][resultingState[1]]
            self.setState(resultingState)
            return resultingState, reward, None, {}
        else:
            reward = self.grid[self.agentPosition[0]][self.agentPosition[1]]
            self.setState(self.agentPosition)
            return self.agentPosition, reward, None, {}

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


class QNetwork(nn.Module):
    """ Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_unit=64, fc2_unit=64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()  ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        print("X")
        print(x)
        print("FC1")
        print(self.fc1(x))
        print("FC2")
        print(self.fc2(x))
        print("FC3")
        print(self.fc3(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
