from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

CONFIG_DIR = Path(__file__).resolve().parents[1] / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(CONFIG_DIR))
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Scale a series to [0, 1]; return zeros if the range collapses."""
    s_min, s_max = series.min(), series.max()
    if pd.isna(s_min) or pd.isna(s_max) or np.isclose(s_max, s_min):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


def find_longest_run(mask: pd.Series) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return start/end timestamps for the longest contiguous True run in a boolean mask."""
    runs: List[Tuple[int, int]] = []
    in_run = False
    start_idx = 0
    for i, flag in enumerate(mask.to_numpy()):
        if flag and not in_run:
            start_idx = i
            in_run = True
        elif not flag and in_run:
            runs.append((start_idx, i - 1))
            in_run = False
    if in_run:
        runs.append((start_idx, len(mask) - 1))
    if not runs:
        return None, None
    longest = max(runs, key=lambda r: r[1] - r[0])
    start_ts = mask.index[longest[0]]
    end_ts = mask.index[longest[1]]
    return start_ts, end_ts


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_market_data.csv"
    figures_dir = Path(__file__).resolve().parents[1] / "figures"

    df = pd.read_csv(data_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)


    if (df["timestamp"].dt.year < 2000).any():
        raise ValueError("Detected timestamps earlier than year 2000; please inspect data integrity.")

   
    if len(df) > 60:
        df = df.tail(60).reset_index(drop=True)
    df_plot = df.copy()

   
    abs_ret = df["close"].pct_change().abs()
    price_signal = abs_ret.rolling(window=5, min_periods=2).mean().fillna(0.0)
    price_signal = minmax_normalize(price_signal)

   
    vol_mean = df["volume"].rolling(window=5, min_periods=3).mean()
    vol_std = df["volume"].rolling(window=5, min_periods=3).std()
    vol_z = ((df["volume"] - vol_mean) / vol_std).abs().fillna(0.0)
    behavioral_signal = minmax_normalize(vol_z)

   
    weight_price, weight_behavioral = 0.5, 0.5
    combined = (weight_price * price_signal) + (weight_behavioral * behavioral_signal)

   
    threshold = 0.65
    mask = combined >= threshold
    window_start, window_end = None, None
    if mask.any():
        runs: List[Tuple[int, int]] = []
        start_idx = None
        for i, flag in enumerate(mask.to_numpy()):
            if flag and start_idx is None:
                start_idx = i
            elif not flag and start_idx is not None:
                runs.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            runs.append((start_idx, len(mask) - 1))
        if runs:
            longest = max(runs, key=lambda r: r[1] - r[0])
            window_start = df_plot.loc[longest[0], "timestamp"]
            window_end = df_plot.loc[longest[1], "timestamp"]

   
    plt.style.use("seaborn-v0_8-whitegrid")
    figures_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "price": "#1f77b4",  
        "behavior": "#ff7f0e",  
        "combined": "#9467bd",  
        "threshold": "#555555",
        "window": "#ffcf70",
    }
    x_min, x_max = df_plot["timestamp"].min(), df_plot["timestamp"].max()

    suspicion_path = figures_dir / "suspicion_window_example.png"
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(df["timestamp"], price_signal, label="Price-based signal", color=colors["price"], linewidth=2)
    ax.plot(
        df["timestamp"],
        behavioral_signal,
        label="Behavioral signal (volume)",
        color=colors["behavior"],
        linewidth=2,
    )
    ax.plot(df["timestamp"], combined, label="Combined suspicion score", color=colors["combined"], linewidth=2.25)
    ax.axhline(threshold, color=colors["threshold"], linestyle="--", linewidth=1.2, label="Threshold")

    if window_start is not None and window_end is not None:
        ax.axvspan(
            window_start,
            window_end,
            color=colors["window"],
            alpha=0.35,
            label="Suspicion window",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized score")
    ax.set_title("Illustrative Multi-Signal Suspicion Window", pad=12)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(x_min, x_max)

 
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate(rotation=15, ha="right")

   
    assert df_plot["timestamp"].dt.year.min() >= 2000, "Plot contains timestamp earlier than 2000."
    x_left = pd.to_datetime(mdates.num2date(ax.get_xlim()[0]))
    assert x_left.date() == x_min.date(), "X-axis lower bound does not match earliest plotted timestamp."

    fig.tight_layout()
    fig.savefig(suspicion_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {suspicion_path}")

   
    interaction_path = figures_dir / "signal_interaction.png"
    fig2, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(10, 9),
        gridspec_kw={"hspace": 0.25},
        constrained_layout=True,
    )

    axes[0].plot(df["timestamp"], price_signal, color=colors["price"], linewidth=2)
    axes[0].set_ylabel("Price signal")
    axes[0].set_title("Price-only view: ambiguous on its own", loc="left", fontsize=11)

    axes[1].plot(df["timestamp"], behavioral_signal, color=colors["behavior"], linewidth=2)
    axes[1].set_ylabel("Behavioral signal")
    axes[1].set_title("Behavior-only view: partial evidence", loc="left", fontsize=11)

    axes[2].plot(df["timestamp"], combined, color=colors["combined"], linewidth=2.25, label="Fused score")
    axes[2].axhline(threshold, color=colors["threshold"], linestyle="--", linewidth=1.2, label="Threshold")
    if window_start is not None and window_end is not None:
        axes[2].axvspan(
            window_start,
            window_end,
            color=colors["window"],
            alpha=0.35,
            label="Suspicion window",
        )
    axes[2].set_ylabel("Fused score")
    axes[2].set_title("Fused view: clearer combined signal", loc="left", fontsize=11)
    axes[2].legend(loc="upper left", frameon=True, framealpha=0.95)

  
    for ax in axes:
        ax.set_ylim(0, 1.0)
        ax.grid(True, which="major", axis="both", alpha=0.2)
        ax.set_xlim(x_min, x_max)

    axes[-1].set_xlabel("Date")

   
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    fig2.autofmt_xdate(rotation=15, ha="right")
    fig2.suptitle("Signal Interaction Illustration", fontsize=13, y=0.995)

   
    assert df_plot["timestamp"].dt.year.min() >= 2000, "Plot contains timestamp earlier than 2000."
    x_left_interaction = pd.to_datetime(mdates.num2date(axes[-1].get_xlim()[0]))
    assert x_left_interaction.date() == x_min.date(), "X-axis lower bound does not match earliest plotted timestamp."

    fig2.savefig(interaction_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {interaction_path}")
    print("Validation: no epoch artifacts detected; all plotted dates are post-2000 and axis bounds align with data.")


if __name__ == "__main__":
    main()
