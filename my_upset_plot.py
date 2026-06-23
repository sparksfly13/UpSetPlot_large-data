"""Plot the candy-basket CSV as an UpSet plot using subset_size='count'.

Reads the boolean indicator CSV produced by generate_candy_data.py, lets
upsetplot aggregate row counts per category combination, and writes a PNG.

Colors are driven by a JSON palette of hexcodes keyed by name (see palette.json).
Recognised keys: background, bars, empty_dots, shading, labels, axis_lines,
grid_lines. Any key you omit falls back to the upsetplot / matplotlib default.

Usage:
    .venv/bin/python plot_candy_upset.py [--csv PATH] [--out PATH] [--palette PATH]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless: write to file instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd

from upsetplot import UpSet, from_indicators

# Columns in the CSV that are NOT category indicators (carried along, ignored).
NON_CATEGORY_COLUMNS = ["Unique code"]


def load_palette(path):
    """Read a JSON palette mapping names to matplotlib colors (e.g. hexcodes).

    Returns an empty dict (use built-in defaults) if the file is absent.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Palette {path} not found; using default colors")
        return {}


def apply_palette(palette):
    """Apply background/label colors via rcParams; return UpSet color kwargs.

    Only keys present in the palette take effect; anything omitted falls back to
    upsetplot / matplotlib defaults. rcParams must be set before the plot is
    built, so call this before constructing/plotting the UpSet.
    """
    if "background" in palette:
        matplotlib.rcParams["axes.facecolor"] = palette["background"]
        matplotlib.rcParams["figure.facecolor"] = palette["background"]
    if "labels" in palette:
        for key in ("text.color", "axes.labelcolor", "xtick.color", "ytick.color"):
            matplotlib.rcParams[key] = palette["labels"]
    # axis_lines colors the spines (the box/frame and x-axis baseline).
    if "axis_lines" in palette:
        matplotlib.rcParams["axes.edgecolor"] = palette["axis_lines"]
    # grid_lines colors the count-axis gridlines upsetplot draws on the bars.
    if "grid_lines" in palette:
        matplotlib.rcParams["grid.color"] = palette["grid_lines"]

    # facecolor drives both the bars and the active ("occupied") matrix dots.
    kwargs = {}
    if "bars" in palette:
        kwargs["facecolor"] = palette["bars"]
    if "empty_dots" in palette:
        kwargs["other_dots_color"] = palette["empty_dots"]
    if "shading" in palette:
        kwargs["shading_color"] = palette["shading"]
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="candy_baskets.csv", help="input CSV path")
    parser.add_argument("--out", default="candy_upset.png", help="output PNG path")
    parser.add_argument(
        "--palette", default="palette.json", help="JSON color palette path"
    )
    args = parser.parse_args()

    color_kwargs = apply_palette(load_palette(args.palette))

    df = pd.read_csv(args.csv)

    # Every column except the reference(s) is a category indicator.
    category_columns = [c for c in df.columns if c not in NON_CATEGORY_COLUMNS]

    # The CSV stores membership as 1/0 integers; convert those columns to
    # booleans so upsetplot reads them as indicators rather than numeric values.
    df[category_columns] = df[category_columns].astype(bool)

    # from_indicators uses only category_columns as the boolean MultiIndex; any
    # other column (the reference) rides along in the data and has no effect on
    # the counts, which are tallied by index.
    indexed = from_indicators(category_columns, data=df)

    # subset_size='count' tallies the number of rows in each combination.
    upset = UpSet(
        indexed,
        orientation="vertical",
        subset_size="count",
        sort_by="cardinality",
        show_counts=True,
        **color_kwargs,
    )
    upset.plot()

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
