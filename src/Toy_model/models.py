import numpy as np
from random import randrange


class FreeEnergyBarrier(object):
    def __init__(self, m, n):
        self.grid = np.zeros((m, n))
        self.m = m
        self.n = n
        self.stateSpace = [i for i in range(self.m * self.n)]
        self.stateSpacePlus = [i for i in range(self.m * self.n)]
        self.actionSpace = {"X": 1, "-X": -1, "Y": self.m, "-Y": -self.m}
        self.possibleActions = ['X', '-X', 'Y', '-Y']
        self.agentPosition = (randrange(self.m), randrange(self.n))

    def getAgentRowAndColumn(self):
        x = self.agentPosition/