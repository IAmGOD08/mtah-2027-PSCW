"""Parse the raw ATD tracking file once and cache it as a compressed .npz.

The raw file is 199,319 lines and takes a minute or two to stream. Caching it
means every later experiment loads in seconds instead of re-parsing.

Coordinates are normalised 0-1 in the raw file and must be scaled to metres
before any distance is meaningful. The pitch is 105.3 x 68.0 m, not square.

Track slots: the metadata lists 46 slots for 22 players. They are tracklets,
not people, and do not follow one player through the match. We keep all 46
and let the team array tell us which side each belongs to.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, require, source_frames  # noqa: E402
import baseline as B  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "tracking.npz")


def build_cache(path=CACHE_PATH, game=GAME, data_dir=None):
    """Read the raw ATD once and save player positions, ball position, team ids.

    Returns the cache path. Safe to call again -- it skips the work if the file
    already exists, unless force=True.
    """
    if os.path.isfile(path):
        print(f"cache already exists: {path}")
        return path

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team = B.team_of_each_track(meta_path)
    n_slots = len(team)

    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    print(f"reading {os.path.basename(raw_path)} ...")
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            idx = int(parts[0]) - 1  # raw frame_id starts at 1, arrays at 0
            if idx < 0 or idx >= n_frames:
                continue

            ball_field = parts[2]
            if ball_field[0] != "N":
                bx, by = ball_field.split(",")[:2]
                ball[idx] = (float(bx) * length, float(by) * width)

            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or group[0] == "N":
                    continue
                px, py = group.split(",")[:2]
                players[idx, slot] = (float(px) * length, float(py) * width)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, players=players, ball=ball, team=team)
    print(f"wrote {path}")
    return path


def load_cache(path=CACHE_PATH):
    """Load the cached tracking. Call build_cache() first if the file is missing."""
    if not os.path.isfile(path):
        build_cache(path)
    z = np.load(path)
    return z["players"], z["ball"], z["team"]


if __name__ == "__main__":
    build_cache()
