#!/usr/bin/env python3
"""Rank ACC classes from 2024 exit survey data (columns L-S)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT = "data/Grad Program Exit Survey Data 2024 (1).xlsx"
UVU_GREEN = "#275D38"
CHART_EXPLANATION = (
    "This graph shows how students ranked each class based on perceived benefit. "
    "Each bar represents one class, and the average score (mean) reflects its overall rank "
    "across survey responses. In this dataset, higher mean values indicate a worse ranking "
    "(i.e., students rated the class as less beneficial), while lower mean values indicate a "
    "better ranking (more beneficial). The classes are ordered from the worst-ranked at the "
    "bottom to the best-ranked at the top, so you can quickly identify which courses students "
    "viewed as least versus most beneficial. Use the bar lengths to compare classes: longer "
    "bars correspond to lower perceived benefit, and shorter bars correspond to higher "
    "perceived benefit."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank ACC classes by mean benefit score from 2024 exit survey data."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help='Path to Excel file (default: "data/Grad Program Exit Survey Data 2024 (1).xlsx")',
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help='Directory for generated artifacts (default: "outputs")',
    )
    return parser.parse_args()


def extract_class_label(cell_value: object) -> str:
    text = "" if pd.isna(cell_value) else str(cell_value).strip()
    match = re.search(r"(ACC\d{4}.*)$", text)
    if match:
        return match.group(1).strip()

    if "-" in text:
        fallback = text.split("-")[-1].strip()
        if fallback:
            return fallback

    return text if text else "Unknown Class"


def make_unique(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        base = label if label else "Unknown Class"
        counts[base] = counts.get(base, 0) + 1
        if counts[base] == 1:
            unique.append(base)
        else:
            unique.append(f"{base} ({counts[base]})")
    return unique


def build_ranking(input_path: Path) -> pd.DataFrame:
    df_raw = pd.read_excel(input_path, header=None, engine="openpyxl")

    header_row = df_raw.iloc[1, 11:19]
    data_block = df_raw.iloc[2:, 11:19]

    labels = make_unique([extract_class_label(value) for value in header_row])
    data_block = data_block.copy()
    data_block.columns = labels

    data = data_block.apply(pd.to_numeric, errors="coerce")

    summary = pd.DataFrame(
        {
            "class": data.columns,
            "mean": data.mean().values,
            "n": data.count().values,
            "median": data.median().values,
            "std": data.std().values,
        }
    )

    zero_n = summary[summary["n"] == 0]
    if not zero_n.empty:
        for class_name in zero_n["class"]:
            print(f"WARNING: Dropping class with no numeric responses: {class_name}")
        summary = summary[summary["n"] > 0].copy()

    if summary.empty:
        raise ValueError("All classes have n == 0 after cleaning. Check source data.")

    summary = summary.sort_values("mean", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))

    return summary


def save_outputs(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "rank_order.csv"
    png_path = output_dir / "rank_order.png"

    summary.to_csv(csv_path, index=False)

    plot_df = summary.sort_values("mean", ascending=True).copy()
    plot_df["label"] = plot_df.apply(lambda r: f"{r['class']} (n={int(r['n'])})", axis=1)

    fig_height = max(6, len(plot_df) * 0.65)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    bars = ax.barh(plot_df["label"], plot_df["mean"], color=UVU_GREEN)

    min_mean_label = plot_df.iloc[0]["label"]

    def get_top_tick_label() -> str | None:
        tick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        if not tick_labels:
            return None
        y0, y1 = ax.get_ylim()
        return tick_labels[-1] if y0 < y1 else tick_labels[0]

    top_tick_label = get_top_tick_label()
    if top_tick_label != min_mean_label:
        ax.invert_yaxis()
        top_tick_label = get_top_tick_label()

    if top_tick_label != min_mean_label:
        raise ValueError("Plot orientation check failed: lowest mean is not at the top.")

    ax.set_xlabel("Mean Rating (1-8)")
    ax.set_ylabel("Class")
    ax.set_xlim(0, 8.6)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, mean_value in zip(bars, plot_df["mean"]):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_value:.2f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_title("2024 Exit Survey: Class Benefit Ranking", fontsize=16, pad=14)

    fig.text(
        0.01,
        0.01,
        CHART_EXPLANATION,
        ha="left",
        va="bottom",
        fontsize=9,
        wrap=True,
    )

    plt.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved chart: {png_path}")


def print_console_summary(summary: pd.DataFrame) -> None:
    top_n = min(3, len(summary))
    print("\nTop classes:")
    for _, row in summary.head(top_n).iterrows():
        print(f"  #{int(row['rank'])}: {row['class']} | mean={row['mean']:.3f}, n={int(row['n'])}")

    print("\nBottom classes:")
    for _, row in summary.tail(top_n).sort_values("mean", ascending=True).iterrows():
        print(f"  #{int(row['rank'])}: {row['class']} | mean={row['mean']:.3f}, n={int(row['n'])}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    summary = build_ranking(input_path)
    save_outputs(summary, output_dir)
    print_console_summary(summary)


if __name__ == "__main__":
    main()
