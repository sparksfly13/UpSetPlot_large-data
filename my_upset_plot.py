"""Plot a boolean-indicator CSV as an UpSet plot using subset_size='count'.

Reads a CSV whose category columns hold 1/0 membership flags, lets upsetplot
aggregate row counts per category combination, and writes two complementary
PNGs per run that partition all intersections at n (--min-subset-size, n > 1): a
"_below<n>" plot (subsets smaller than n) and a "_min<n>" plot (subsets of at
least n). Columns that are not category indicators (e.g. an id) are listed in
NON_CATEGORY_COLUMNS and carried along but ignored when tallying.

Colors are driven by a JSON palette of hexcodes keyed by name (see palette.json).
Recognised keys: background, bars, empty_dots, shading, labels, axis_lines,
grid_lines, category_labels, category_label_size. category_labels is one color
for all category names or an object mapping category name -> color (unlisted
names keep the "labels" color); category_label_size is a font size in pt. Any
key you omit falls back to the upsetplot / matplotlib default.

Usage:
    .venv/bin/python my_upset_plot.py [--csv PATH] [--out PATH] [--palette PATH]
        [--sep CHAR] [--min-subset-size N]
"""

import argparse
import json
import os

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


def suffixed(path, suffix):
    """Insert _<suffix> before the file extension of path."""
    root, ext = os.path.splitext(path)
    return f"{root}_{suffix}{ext}"


def render_upset(
    indexed,
    color_kwargs,
    size_kwargs,
    out_path,
    freq_xlim=None,
    label_colors=None,
    label_size=None,
):
    """Build one UpSet plot with the given size filter and save it to out_path.

    size_kwargs is passed straight to UpSet, e.g. {"min_subset_size": 100} or
    {"max_subset_size": 99}. If freq_xlim is given, the Frequency (intersection
    size) axis is pinned to those limits instead of autoscaling. label_colors
    (one color, or a dict of category name -> color) and label_size (pt) style
    the category name labels. Returns the Frequency-axis limits actually used,
    so a caller can share one plot's scale with another.
    """
    # Column width (element_size, pt) is fixed and does not follow the label
    # font, so widen columns for large labels to keep neighbours from touching.
    # 32 is upsetplot's default.
    element_size = max(32, 2 * label_size) if label_size else 32
    upset = UpSet(
        indexed,
        orientation="vertical",
        subset_size="count",
        sort_by="cardinality",
        sort_categories_by="input",  # keep CSV column order, not by total count
        show_counts=True,
        element_size=element_size,
        **size_kwargs,
        **color_kwargs,
    )
    # upsetplot reserves room for the category labels from xtick.labelsize at
    # layout time, so the label size must be in effect while plotting. It
    # measures in whole-figure units but the grid only fills the subplot
    # fraction of the figure height, so inflate the layout size to compensate
    # (otherwise long labels run into the totals bars).
    rc = {}
    if label_size:
        fill = (
            matplotlib.rcParams["figure.subplot.top"]
            - matplotlib.rcParams["figure.subplot.bottom"]
        )
        rc["xtick.labelsize"] = label_size / fill
    with matplotlib.rc_context(rc):
        axes = upset.plot()
    if label_size:
        axes["matrix"].tick_params(labelsize=label_size)
        # The Frequency axis also uses xtick.labelsize; restore its default.
        axes["intersections"].tick_params(
            axis="x", labelsize=matplotlib.rcParams["xtick.labelsize"]
        )
    if isinstance(label_colors, dict):
        for label in axes["matrix"].get_xticklabels():
            if label.get_text() in label_colors:
                label.set_color(label_colors[label.get_text()])
    elif label_colors:
        axes["matrix"].tick_params(labelcolor=label_colors)
    # Rename the intersection bar chart's default "Intersection size" label.
    # In this vertical layout that label sits on the intersections x-axis.
    axes["intersections"].set_xlabel("Frequency")
    if freq_xlim is not None:
        axes["intersections"].set_xlim(freq_xlim)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # Read limits after savefig so autoscaling has been finalized by the draw.
    used_xlim = axes["intersections"].get_xlim()
    plt.close("all")  # free the figure before rendering the next plot
    print(f"Wrote {out_path}")
    return used_xlim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data.csv", help="input CSV path")
    parser.add_argument(
        "--out",
        default="my_upset_plot.png",
        help="base output PNG path; written suffixed as _below<n> and _min<n>",
    )
    parser.add_argument(
        "--palette", default="palette.json", help="JSON color palette path"
    )
    parser.add_argument("--sep", default=",", help="CSV field delimiter")
    parser.add_argument(
        "--min-subset-size",
        type=int,
        default=2,
        help="min members for the filtered (_min<n>) plot; n > 1",
    )
    args = parser.parse_args()

    palette = load_palette(args.palette)
    color_kwargs = apply_palette(palette)
    label_kwargs = {
        "label_colors": palette.get("category_labels"),
        "label_size": palette.get("category_label_size"),
    }

    df = pd.read_csv(args.csv, sep=args.sep)

    # Every column except the reference(s) is a category indicator.
    category_columns = [c for c in df.columns if c not in NON_CATEGORY_COLUMNS]

    # The CSV stores membership as 1/0 integers; convert those columns to
    # booleans so upsetplot reads them as indicators rather than numeric values.
    df[category_columns] = df[category_columns].astype(bool)

    # from_indicators uses only category_columns as the boolean MultiIndex; any
    # other column (the reference) rides along in the data and has no effect on
    # the counts, which are tallied by index.
    indexed = from_indicators(category_columns, data=df)

    # Render two complementary plots that partition all intersections at n:
    #   _below<n>: subsets smaller than n   (max_subset_size = n - 1)
    #   _min<n>:   subsets of at least n     (min_subset_size = n)
    n = args.min_subset_size
    if n <= 1:
        print(f"--min-subset-size is {n}; the _below{n} plot will be empty")
    # Render _min first, then reuse its Frequency-axis scale for _below so the
    # two plots share an identical (comparable) horizontal scale.
    freq_xlim = render_upset(
        indexed,
        color_kwargs,
        {"min_subset_size": n},
        suffixed(args.out, f"min{n}"),
        **label_kwargs,
    )
    render_upset(
        indexed,
        color_kwargs,
        {"max_subset_size": n - 1},
        suffixed(args.out, f"below{n}"),
        freq_xlim=freq_xlim,
        **label_kwargs,
    )


if __name__ == "__main__":
    main()
