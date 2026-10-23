"""Cache V-JEPA 2 feature maps of full frames for LaSOT videos (computed once, reused by every experiment).

For every `--every`-th frame t a causal clip (frames t-30, t-28, ..., t with the defaults) is encoded
and the 16 x 16 x 1024 map of its last time step is stored. Frames are streamed from HuggingFace.

    python scripts/cache_vjepa.py --sequences configs/splits/dev_occlusion.txt
    python scripts/cache_vjepa.py --sequences ... --shard 0 --num-shards 2   # one process per GPU

With --surprise the V-JEPA predictor also predicts the last slot from the earlier ones and the
per-token surprise 1 - cos(predicted, real) is stored (src/vjepa/predictor.py); ~2x slower.

Output: <out>/<seq>.npz with `frames` (int, 0-based frame index t), `maps` (float16, N x 16 x 16 x C)
and with --surprise `surprise` (float16, N x 16 x 16).
Size: ~0.5 MB per map, i.e. ~180 MB per average dev video with --every 8.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.lasot import read_split  # noqa: E402
from src.data.remote_zip import stream_sequences  # noqa: E402
from src.vjepa.encoder import DEFAULT_MODEL, VJEPAEncoder, clip_indices  # noqa: E402
from src.vjepa.predictor import VJEPAPredictor  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", type=Path, default=REPO_ROOT / "configs" / "splits" / "dev_occlusion.txt")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--every", type=int, default=8)
    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--batch", type=int, default=4, help="clips per forward pass")
    p.add_argument("--surprise", action="store_true", help="also store the predictor's surprise map")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "cache" / "vjepa")
    p.add_argument("--work", type=Path, default=REPO_ROOT / "tmp_frames")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()

    seqs = sorted(read_split(a.sequences))[a.shard::a.num_shards]
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seqs if not (a.out / f"{s}.npz").exists()]
    print(f"shard {a.shard}/{a.num_shards}: {len(seqs)} videos, {len(todo)} to encode", flush=True)
    if not todo:
        return
    enc = VJEPAPredictor(a.model) if a.surprise else VJEPAEncoder(a.model)
    t0, n_maps = time.time(), 0
    for k, (seq, seq_dir, err) in enumerate(stream_sequences(todo, a.work / f"vjepa_shard{a.shard}"), 1):
        if err is not None:
            print(f"[{k}/{len(todo)}] {seq}: download FAILED ({err})", flush=True)
            continue
        paths = sorted((seq_dir / "img").glob("*.jpg"))
        ts = np.arange(0, len(paths), a.every)
        cache: dict[int, np.ndarray] = {}  # consecutive clips share most frames

        def frame(i):
            if i not in cache:
                cache[i] = cv2.cvtColor(cv2.imread(str(paths[i])), cv2.COLOR_BGR2RGB)
            return cache[i]

        maps, surprise = [], []
        for b in range(0, len(ts), a.batch):
            clips = [[frame(i) for i in clip_indices(t, a.clip_len, a.stride)] for t in ts[b:b + a.batch]]
            if a.surprise:
                m, s = enc.encode_with_surprise(clips)
                maps.append(m.numpy())
                surprise.append(s.numpy())
            else:
                maps.append(enc.encode_batch(clips).numpy())
            keep = ts[b] - a.stride * a.clip_len  # frames older than the next clip can go
            for i in [i for i in cache if i < keep]:
                del cache[i]
        extra = dict(surprise=np.concatenate(surprise)) if a.surprise else {}
        np.savez(a.out / f"{seq}.npz", frames=ts, maps=np.concatenate(maps), **extra)
        shutil.rmtree(seq_dir, ignore_errors=True)
        n_maps += len(ts)
        el = time.time() - t0
        print(f"[{k}/{len(todo)}] {seq}: {len(ts)} maps | {n_maps / el:.1f} maps/s | elapsed {el / 60:.0f} min, "
              f"eta {el / k * (len(todo) - k) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
