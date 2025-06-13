#!/usr/bin/env python3
"""
Create a dummy Neuropixels `continuous.dat` file that matches the
timestamps/sample count already on disk, but carries data for just one
(real) channel.  All other channels are zeros, keeping the interleaved
layout expected by Open Ephys.
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stream_dir", help="path to the stream folder that contains timestamps.npy")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random-generator seed (omit for non-reproducible values)",
    )
    ap.add_argument(
        "--active_channel",
        type=int,
        default=0,
        help="index (0-based) of the channel that will receive random data",
    )
    args = ap.parse_args()

    stream_dir = Path(args.stream_dir).resolve()
    ts_file = stream_dir / "timestamps.npy"
    sn_file = stream_dir / "sample_numbers.npy"
    if not ts_file.exists() or not sn_file.exists():
        raise FileNotFoundError("timestamps.npy or sample_numbers.npy not found in the specified folder")

    n_samples = np.load(sn_file, mmap_mode="r").shape[0]

    rng = np.random.default_rng(args.seed)

    # Allocate (samples, channels) and fill one column with random int16
    data = np.zeros((n_samples, 1), dtype="<i2")  # little-endian int16
    data[:, 0] = rng.integers(
        low=-32768, high=32767, size=n_samples, dtype=np.int16
    )

    # Write as sample-interleaved binary
    out_path = stream_dir / "continuous.dat"
    data.T.ravel().tofile(out_path)
    print(f"Wrote {out_path}  ({n_samples} samples × {1} channel)")


if __name__ == "__main__":
    main()