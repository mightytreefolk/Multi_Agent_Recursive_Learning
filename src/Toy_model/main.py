import numpy as np
import pandas as pd
import uuid
from datetime import datetime
import os
import matplotlib.pyplot as plt
from models import FreeEnergyBarrier


def maxAction(Q, state, actions):
    values = np.array([Q[(state[0], state[1]), a] for a in actions])
    action = np.argmax(values)
    print(actions[action])
    return actions[action]

if __name__ == '__main__':
    x = 200
    y = 200
    env = FreeEnergyBarrier(x, y)
    # model hyperparameters
    ALPHA = 0.1 # Learning Rate
    GAMMA = 0.9 # discount factor
    EPS = 1.0  # Epsilon greedy action selection

    Q = {}
    for state in env.stateSpace[0]:
        for i in env.stateSpace[1]:
            for action in env.possibleActions:
                Q[(state, i), action] = 0

    numGames = 5
    totalRewards = np.zeros(numGames)
    env.render()
    df = pd.DataFrame()
    for i in range(numGames):
        print("Starting new run")
        done = False
        epRewards = 0
        observation = env.reset()
        t = 0
        dt = .01
        run = []
        while not done:
            rand = np.random.random()
            action = maxAction(Q, observation, env.possibleActions) if rand < (1-EPS) else env.actionSpaceSample()
            observation_, reward, done, info = env.step(action)
            epRewards += reward

            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[(observation[0], observation[1]), action] = Q[(observation[0], observation[1]), action] + \
                                                          ALPHA*(reward + GAMMA * Q[(observation_[0], observation_[1]), action_] - Q[(observation[0], observation[1]), action])
            observation = observation_
            run.append(observation)
            t = t + dt
        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards

        run_name = str(uuid.uuid4())
        df[run_name] = run
        run.clear()

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    directory = "Dir_with_runs-{time}".format(time=timestamp)
    os.mkdir(directory)
    path = os.path.join(directory, "Run.csv")
    grid_path = os.path.join(directory, "Grid.csv")
    df.to_csv(path, index=False, encoding='utf-8', sep='\t',)
    grid = env.makeGrid(x, y)
    grid.to_csv(grid_path, index=False, encoding='utf-8', sep='\t',)

    plt.plot(totalRewards)
    plt.show()

