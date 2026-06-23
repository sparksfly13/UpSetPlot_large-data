"""Plot the candy-basket CSV as an UpSet plot using subset_size='count'.

Reads the boolean indicator CSV produced by generate_candy_data.py, lets
upsetplot aggregate row counts per category combination, and writes a PNG.

Usage:
    .venv/bin/python plot_candy_upset.py [--csv PATH] [--out PATH]
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: write to file instead of opening a window
import matplotlib.pyplot as plt
import pandas as pd

from upsetplot import UpSet, from_indicators

# Columns in the CSV that are NOT category indicators (carried along, ignored).
NON_CATEGORY_COLUMNS = ["Purchase reference"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="candy_baskets.csv", help="input CSV path")
    parser.add_argument("--out", default="candy_upset.png", help="output PNG path")
    args = parser.parse_args()

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
    )
    upset.plot()

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
