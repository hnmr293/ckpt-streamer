import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def main():
    csv_path = os.path.dirname(__file__) + "/perf.csv"
    df = pd.read_csv(csv_path, header=None, names=["name", "numel", "time"])

    stats = df.groupby(["name", "numel"])["time"].agg(["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])

    names = [
        "w/o ckpt_streamer",
        "w/ ckpt_streamer",
    ]

    fig = go.Figure()

    for name in names:
        group_data = stats[stats["name"] == name]
        if len(group_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=group_data["numel"],
                    y=group_data["mean"],
                    mode="lines+markers",
                    name=name,
                    error_y=dict(type="data", array=group_data["se"], visible=True),
                )
            )

    fig.update_layout(
        title="Performance",
        xaxis_title="Number of elements",
        yaxis_title="process time (s)",
    )

    fig.write_image(os.path.dirname(__file__) + "/perf.png")


if __name__ == "__main__":
    main()
