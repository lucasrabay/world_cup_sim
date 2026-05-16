"""Save the current baseline predictions as a timestamped snapshot.

Use this once before kickoff to lock in the pre-tournament predictions. After
the tournament you can ``git log`` the snapshots / diff them to write the
retrospective post.

Usage::

    python scripts/snapshot_predictions.py             # uses today's date
    python scripts/snapshot_predictions.py --tag pre-wc

Reads ``simulation/results/baseline.parquet`` and writes:

* ``simulation/results/snapshots/snapshot_YYYY-MM-DD[_tag].csv``
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import SIM_RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot current WC predictions.")
    parser.add_argument("--source", default=str(SIM_RESULTS / "baseline.parquet"),
                        help="Parquet to snapshot (defaults to the baseline run).")
    parser.add_argument("--tag", default=None,
                        help="Optional suffix for the snapshot filename.")
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source frame not found: {src}")
    df = pd.read_parquet(src)

    snap_dir = SIM_RESULTS / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    today = _date.today().isoformat()
    name = f"snapshot_{today}" + (f"_{args.tag}" if args.tag else "") + ".csv"
    out = snap_dir / name
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} teams → {out}")

    # Cheap human summary so you can sanity-check at a glance.
    top = df.sort_values("p_champion", ascending=False).head(10)
    print("\nTop 10 champion probabilities at snapshot:")
    print(f"  {'Team':<28} {'p_champion':>11}")
    for _, row in top.iterrows():
        print(f"  {row['team']:<28} {row['p_champion']*100:10.2f}%")


if __name__ == "__main__":
    main()
