"""Generate a CSV of random candy "baskets" for UpSet plotting.

Each row is one observation (e.g. a purchase). Each of the 5 candy columns is a
boolean indicating whether that candy is present in the basket. With many rows,
the same membership combinations recur many times -- which is exactly what
upsetplot's ``subset_size='count'`` aggregates: it tallies how many rows fall
into each of the 2**5 = 32 possible category combinations.

Usage:
    .venv/bin/python generate_candy_data.py [--rows N] [--seed S] [--out PATH]
"""

import argparse
import uuid

import numpy as np
import pandas as pd

# Non-category column: a unique reference identifying each row. It rides along in
# the CSV but must have no effect on the UpSet graph.
REFERENCE_COLUMN = "Purchase reference"

# Deliberately varied names, including some long ones, to stress-test how
# upsetplot lays out category labels.
CATEGORIES = [
    "M&Ms",
    "Reese's Peanut Butter Cups",
    "Snickers",
    "Toblerone",
    "Ferrero Rocher",
    "Bounty",
]

# Per-candy probability of appearing in a basket. Different rates make the
# totals (and intersections) vary, so the plot is more interesting than uniform.
PROBABILITIES = {
    "M&Ms": 0.55,
    "Reese's Peanut Butter Cups": 0.40,
    "Snickers": 0.45,
    "Toblerone": 0.20,
    "Ferrero Rocher": 0.15,
    "Bounty": 0.30,
}


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = {
        name: rng.random(rows) < PROBABILITIES[name] for name in CATEGORIES
    }
    df = pd.DataFrame(columns)
    # Seeded UUIDs so the whole CSV stays reproducible for a given seed.
    raw = rng.integers(0, 256, size=(rows, 16), dtype=np.uint8)
    df.insert(0, REFERENCE_COLUMN, [str(uuid.UUID(bytes=bytes(row))) for row in raw])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000, help="number of observations")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    parser.add_argument("--out", default="candy_baskets.csv", help="output CSV path")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    # Store membership as 1/0 rather than True/False; the plotting script
    # converts these back to booleans. The reference column stays a string.
    out = df.copy()
    out[CATEGORIES] = out[CATEGORIES].astype(int)
    out.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} rows x {len(CATEGORIES)} categories (+1 reference) to {args.out}")
    print("\nPer-category totals (rows where present):")
    print(df[CATEGORIES].sum().to_string())
    print(f"\nDistinct combinations present: {df[CATEGORIES].value_counts().shape[0]} of {2 ** len(CATEGORIES)}")


if __name__ == "__main__":
    main()
