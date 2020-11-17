import numpy as np
import pandas as pd
import uuid
from datetime import datetime
import os

from models import FreeEnergyBarrier, Agent

if __name__ == '__main__':
    LR = 0.001
    GAMMA = 0.99
    BATCH_SIZE = 64
    NUMBER_OF_ACTIONS = 4
    EPSILON = 1.0
    env = FreeEnergyBarrier(200, 200)
    agent = Agent(gamma=GAMMA, epsilon=EPSILON, batch_size=BATCH_SIZE,
                  n_actions=NUMBER_OF_ACTIONS, eps_end=0.01, input_dims=[2], lr=LR)
    scores, eps_history = [], []
    n_runs = 25
    df = pd.DataFrame()
    for i in range(n_runs):
        score = 0
        done = False
        observation = env.reset()
        run = []
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
            run.append(observation)

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        run_name = str(uuid.uuid4())
        x = pd.DataFrame()
        x[run_name] = run
        df = pd.concat([df, x], axis=1)
        run.clear()

        print("Episode: ", i,
              "Score: {}".format(score),
              "Average score: {}".format(avg_score),
              "Epsilon {}".format(agent.epsilon))

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    directory = "Dir_with_runs-{time}".format(time=timestamp)
    os.mkdir(directory)
    path = os.path.join(directory, "Run.csv")
    grid_path = os.path.join(directory, "Grid.csv")
    df.to_csv(path, index=False, encoding='utf-8', sep='\t',)
    grid = env.makeGrid(200, 200)
    grid.to_csv(grid_path, index=False, encoding='utf-8', sep='\t',)