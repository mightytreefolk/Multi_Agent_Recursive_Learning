import numpy as np
from random import randrange
import gym
from scipy.interpolate import griddata
import pandas as pd


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

    def step(self, action: str):
        resultingState = []
        for i, j in zip(self.agentPosition, self.actionSpace[action]):
            resultingState.append(i+j)
        # resultingState = np.array(resultingState)
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