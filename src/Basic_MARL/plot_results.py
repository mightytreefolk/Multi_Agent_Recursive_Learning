import pandas as pd
import plotly.graph_objects as go
import os
import ast


def plot_run(dir):
    path = os.path.join(dir, "Run.csv")
    df = pd.read_csv(path, sep='\t')
    return df

def plot_grid(dir):
    path = os.path.join(dir, "Grid.csv")
    df = pd.read_csv(path, sep='\t')
    return df.T

def main():
    print("What directory would you like to explore?")
    dir = input("Directory name: ")
    grid_df = plot_grid(dir=dir)
    run_df = plot_run(dir=dir)

    fig = go.Figure(data=go.Contour(z=grid_df))
    columns = list(run_df)
    x = []
    y = []
    for i in columns:
        for j in run_df[i]:
            j = ast.literal_eval(j)
            x.append(j[1])
            y.append(j[0])

        run = go.Scatter(x=x, y=y)
        fig.add_trace(run)
        x.clear()
        y.clear()

    fig.show()


if __name__ == '__main__':
    main()