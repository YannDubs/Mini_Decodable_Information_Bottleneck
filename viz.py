import os
from glob import glob
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "For plotting you need `pip install seaborn, pandas, matplotlib`"
    )


def plot_results(name):
    """Plot the results from the 1player, 2player_avg, 2player_worst experiments for the desired run `name`."""

    # all the test log loss
    test_df = pd.read_csv(f"results/{name}.csv").drop("acc", axis=1)
    test_df.rename(columns={"loss": "value"}, inplace=True)
    test_df["loss"] = "test"

    # all the train losses
    train_dfs = []
    for f in glob(f"logs/csv/{name}/**/metrics.csv", recursive=True):
        # select only last epoch train_loss (average)
        train_df = pd.read_csv(f).groupby("epoch").mean().sort_index().iloc[-1:]

        if "1player" in f:
            # we want to plot log loss not the actual DIB loss
            train_df["train_loss"] = train_df["H_V_yCz"]

        train_df = train_df.loc[:, ["train_loss"]]

        # keys for merging
        train_df["mode"] = (
            f.split("/")[3]
            .replace("worst", "2player_worst")
            .replace("avg", "2player_avg")
        )
        train_df["beta"] = float(f.split("beta")[-1].split("_")[0])
        train_df["seed"] = int(f.split("seed")[-1].split("/")[0])
        train_df.rename(columns={"train_loss": "value"}, inplace=True)
        train_df["loss"] = "train"

        # only average last epoch
        train_dfs.append(train_df)

    train_dfs = pd.concat(train_dfs, axis=0).reset_index(drop=True)

    # merge all
    df = pd.concat([test_df, train_dfs], axis=0).reset_index(drop=True)

    # plot
    sns_plot = sns.relplot(
        data=df,
        x="beta",
        y="value",
        hue="loss",
        facet_kws={"sharey": False},
        kind="line",
        legend="full",
        col="mode",
    )
    plt.xscale(
        value="symlog",
        base=10,
        subs=list(range(10)),
        linthresh=df.beta[df.beta != 0].min(),
    )
    sns_plot.set(xlim=(0, None))
    plt.tight_layout()

    # save
    sns_plot.fig.savefig(os.path.join("results", f"{name}.png"), dpi=300)
    plt.close(sns_plot.fig)


if __name__ == "__main__":
    plot_results(sys.argv[1])
