"""Dead-ball detection from tracking evidence.

Signals used:
1. Ball outside pitch bounds (with a small margin for tracker noise).
2. Ball speed near zero for a sustained period.
3. Player speed low (walking, not running) as a secondary check.

Smart Tagging set pieces are used as a cross-check, not as the primary
signal, because they are inferred and can be wrong.

Output: a boolean array, True where the frame is likely dead ball.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, asset_paths  # noqa: E402
from mtah.cache import load_cache  # noqa: E402

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]

# Ball outside these bounds (in metres, with margin) is out of play.
MARGIN = 2.0
X_MIN, X_MAX = -MARGIN, LENGTH + MARGIN
Y_MIN, Y_MAX = -MARGIN, WIDTH + MARGIN

# Ball speed below this (m/s) for at least this many frames = likely dead ball.
BALL_SPEED_THRESHOLD = 1.5
BALL_STILL_FRAMES = 25  # 1 second

# Player mean speed below this (m/s) suggests walking, not running.
PLAYER_SPEED_THRESHOLD = 2.0
PLAYER_STILL_FRAMES = 50  # 2 seconds


def ball_speed(ball, fps=FPS):
    """Per-frame ball speed in m/s. NaN where ball is missing or jump is implausible."""
    v = np.full(len(ball), np.nan, dtype=np.float32)
    dt = 1.0 / fps
    for i in range(1, len(ball)):
        if np.isfinite(ball[i]).all() and np.isfinite(ball[i - 1]).all():
            d = np.linalg.norm(ball[i] - ball[i - 1])
            # Reject impossible jumps: ball can't move more than ~40 m/s realistically
            if d / dt < 40.0:
                v[i] = d / dt
    return v


def player_mean_speed(players, fps=FPS):
    """Per-frame mean speed across all visible players, in m/s."""
    n = len(players)
    v = np.full(n, np.nan, dtype=np.float32)
    dt = 1.0 / fps
    for i in range(1, n):
        prev, curr = players[i - 1], players[i]
        ok = np.isfinite(prev).all(axis=1) & np.isfinite(curr).all(axis=1)
        if ok.sum() < 5:  # need at least 5 visible players to trust the mean
            continue
        d = np.linalg.norm(curr[ok] - prev[ok], axis=1)
        v[i] = d.mean() / dt
    return v


def out_of_bounds(ball):
    """True where ball is outside pitch bounds (with margin)."""
    x, y = ball[:, 0], ball[:, 1]
    return (x < X_MIN) | (x > X_MAX) | (y < Y_MIN) | (y > Y_MAX)


def sustained(mask, min_frames):
    """True where mask has been True for at least min_frames consecutive frames."""
    out = np.zeros_like(mask, dtype=bool)
    count = 0
    for i, m in enumerate(mask):
        count = count + 1 if m else 0
        if count >= min_frames:
            out[i] = True
    return out


def detect(players, ball, use_smart=True):
    """Combine signals into a dead-ball mask."""
    bs = ball_speed(ball)
    ps = player_mean_speed(players)

    # Signal 1: ball out of bounds
    oob = out_of_bounds(ball)

    # Signal 2: ball still for a sustained period
    ball_still = sustained(
        np.isfinite(bs) & (bs < BALL_SPEED_THRESHOLD), BALL_STILL_FRAMES
    )

    # Signal 3: players walking (secondary, only when ball is also still or missing)
    players_slow = sustained(
        np.isfinite(ps) & (ps < PLAYER_SPEED_THRESHOLD), PLAYER_STILL_FRAMES
    )

    # Combine: out of bounds OR (ball still AND players slow)
    dead = oob | (ball_still & players_slow)

    # Optional: cross-check with Smart Tagging set pieces
    if use_smart:
        smart = smart_tagging_dead()
        if smart is not None:
            dead = dead | smart

    return dead


def smart_tagging_dead():
    """Load Smart Tagging set-piece intervals and mark them as dead ball.

    Returns a boolean array over video frames, or None if the file is missing.
    Intervals are in seconds from video start; convert to frames with round(s * 25).
    """
    try:
        import xml.etree.ElementTree as ET

        path = asset_paths()["smart_tagging"]
        if not os.path.isfile(path):
            return None
        tree = ET.parse(path)
        root = tree.getroot()
        n = GAMES[GAME]["n_video_frames"]
        mask = np.zeros(n, dtype=bool)
        for instance in root.iter("instance"):
            code = instance.findtext("code", default="").upper()
            if "SET PIECE" not in code:
                continue
            start = instance.findtext("start")
            end = instance.findtext("end")
            if start is None or end is None:
                continue
            s = int(round(float(start) * FPS))
            e = int(round(float(end) * FPS))
            mask[max(0, s) : min(n, e + 1)] = True
        return mask
    except Exception:
        return None


if __name__ == "__main__":
    players, ball, team = load_cache()
    dead = detect(players, ball)
    print(f"dead-ball frames: {dead.sum()} of {len(dead)} ({100 * dead.mean():.1f}%)")
    # Count stretches
    changes = np.flatnonzero(np.diff(dead.astype(int), prepend=0))
    runs = np.split(np.arange(len(dead)), changes[1:])
    d_runs = [r for r in runs if dead[r[0]]]
    lengths = np.array([len(r) for r in d_runs], dtype=float)
    print(f"dead-ball stretches: {len(d_runs)}")
    print(f"total dead-ball time: {lengths.sum() / FPS:.1f} s")
    print(
        f"mean stretch: {lengths.mean() / FPS:.1f} s" if len(d_runs) else "no stretches"
    )
