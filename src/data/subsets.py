"""Dev and train subsets of LaSOT."""
from __future__ import annotations

import pandas as pd

from src.data.lasot import Sequence
from src.eda.stats import runs


def occlusion_profile(seqs: list[Sequence], min_len: int = 6) -> pd.DataFrame:
    """Per video: absence episodes the target returns from, and how many last >= min_len frames."""
    rows = []
    for s in seqs:
        starts, lengths = runs(s.absent)
        back = [(a, l) for a, l in zip(starts, lengths) if a > 0 and a + l < s.n_frames]
        rows.append(dict(seq=s.name, cls=s.cls, n_frames=s.n_frames, frac_absent=s.absent.mean(),
                         n_episodes=len(back), n_long=sum(l >= min_len for _, l in back),
                         n_very_long=sum(l >= 30 for _, l in back)))
    return pd.DataFrame(rows)


def train_subset(profile: pd.DataFrame, per_class: int = 4) -> pd.DataFrame:
    """`per_class` videos per class, videos with occlusions first."""
    ranked = profile.sort_values(["n_long", "n_episodes", "frac_absent"], ascending=False)
    return ranked.groupby("cls").head(per_class).sort_values("seq", key=lambda s: s.str.lower())


def dev_occlusion_subset(profile: pd.DataFrame, n: int = 40, max_per_class: int = 1) -> pd.DataFrame:
    """Most occlusion-dense videos, at most `max_per_class` per class."""
    ranked = profile[profile.n_long > 0].sort_values(["n_very_long", "n_long", "frac_absent"], ascending=False)
    picked = ranked.groupby("cls").head(max_per_class).head(n)
    return picked.sort_values("seq", key=lambda s: s.str.lower())
