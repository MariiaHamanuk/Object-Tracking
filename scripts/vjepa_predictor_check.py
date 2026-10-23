"""Does the V-JEPA 2 predictor (world model) help tracking? Two probes, no training, no OSTrack.

Every `--every`-th frame t of each dev video, causal clip ending at t (16 frames, stride 2 -> 8 time
slots of 2 frames; the last slot = "now"). m = target memory pooled in the GT box of frame 0.

A. Surprise = occlusion detector.
   Hide the tokens of the last slot inside the last box where the target was visible (what a tracker
   that froze there would watch), predict them from the rest of the clip, compare with what the
   encoder actually sees there:
     surprise    = 1 - cos(predicted, real)          (pooled over the hidden tokens)
     mem_drop    = 1 - cos(real, m)                  (encoder-only baseline)
     mem_drop_pred = 1 - cos(predicted, m)
   If the target was there a moment ago and now is hidden, the predictor should "expect" it while the
   encoder sees the occluder -> surprise up. Scored by AUROC of separating absent (full occlusion or
   out of view at t) from visible frames.

B. Future prediction = where will the target be.
   Context = the first `--context-slots` slots only, predict all tokens of the remaining slots, and
   search m on the predicted last slot. Compared with the real last slot (upper bound) and the last
   context slot (no prediction, "the scene as last seen"):
     hit = most similar token lies in the GT box at t; also the GT percentile (50 = chance).
   Only frames with the target visible at t.

    python scripts/vjepa_predictor_check.py --sequences configs/splits/dev_occlusion.txt --out results/vjepa_predictor_check
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.lasot import load_sequence, read_split  # noqa: E402
from src.data.remote_zip import stream_sequences  # noqa: E402
from src.vjepa.encoder import clip_indices, pool_box  # noqa: E402
from src.vjepa.predictor import VJEPAPredictor  # noqa: E402


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0))


def search(m: torch.Tensor, tokens: torch.Tensor, gt_idx: np.ndarray) -> tuple[bool, float]:
    """tokens: (256, C) one slot. Is the best match inside the GT tokens, and the GT percentile."""
    sim = F.cosine_similarity(tokens, m[None], dim=-1)
    s_gt = float(sim[gt_idx].max())
    return bool(int(sim.argmax()) in set(gt_idx.tolist())), float((sim < s_gt).float().mean() * 100)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties count half."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()[: len(pos)]
    return float((r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(df: pd.DataFrame) -> dict:
    out = {"n_frames": len(df), "n_videos": int(df.seq.nunique()), "n_absent": int(df.absent.sum())}
    a = df[df.target_seen_recently]  # A only makes sense right after the target was visible
    lab = a.absent.to_numpy(bool)
    out["A_occlusion_auroc"] = {k: auroc(a[k].to_numpy(), lab) for k in ("surprise", "mem_drop", "mem_drop_pred")}
    out["A_n"] = {"frames": len(a), "absent": int(lab.sum())}
    v = df[~df.absent]
    out["B_future"] = {k: {"top1_hit": float(v[f"hit_{k}"].mean() * 100),
                           "gt_percentile": float(v[f"pct_{k}"].mean())}
                       for k in ("real", "predicted", "last_seen")}
    out["B_future"]["chance_top1"] = float(v.gt_share.mean() * 100)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", default=str(REPO_ROOT / "configs" / "splits" / "dev_occlusion.txt"))
    p.add_argument("--every", type=int, default=16)
    p.add_argument("--context-slots", type=int, default=4, help="B: visible slots (of 8), the rest is predicted")
    p.add_argument("--recent", type=int, default=30, help="A: target visible within this many frames before t")
    p.add_argument("--max-frames", type=int, default=None, help="only for quick tests")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "vjepa_predictor_check")
    p.add_argument("--work", type=Path, default=REPO_ROOT / "tmp_frames")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()

    seqs = sorted(read_split(a.sequences)) if Path(a.sequences).is_file() else a.sequences.split(",")
    seqs = seqs[a.shard::a.num_shards]
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seqs if not (a.out / f"{s}.csv").exists()]
    print(f"shard {a.shard}/{a.num_shards}: {len(seqs)} videos, {len(todo)} to run", flush=True)
    enc = VJEPAPredictor() if todo else None
    t0 = time.time()
    for k, (seq, seq_dir, err) in enumerate(stream_sequences(todo, a.work / f"pred_shard{a.shard}"), 1):
        if err is not None:
            print(f"[{k}/{len(todo)}] {seq}: download FAILED ({err})", flush=True)
            continue
        s = load_sequence(seq_dir, read_image_size=False)
        paths = sorted((seq_dir / "img").glob("*.jpg"))[: a.max_frames]
        frames: dict[int, np.ndarray] = {}

        def frame(i):
            if i not in frames:
                frames[i] = cv2.cvtColor(cv2.imread(str(paths[i])), cv2.COLOR_BGR2RGB)
            return frames[i]

        h, w = frame(0).shape[:2]
        g, last = enc.grid, 8 - 1                                  # 16 frames / tubelet 2 = 8 slots
        last_tok = enc.slot_tokens(last)
        valid = s.valid_box
        first = enc.encode_tokens(enc.preprocess([frame(i) for i in clip_indices(0)]))[last_tok]
        m = F.normalize(pool_box(first.view(g, g, -1), s.boxes[0], w, h), dim=0)
        ctx_b = np.arange(a.context_slots * g * g)
        tgt_b = np.arange(a.context_slots * g * g, 8 * g * g)

        rows = []
        for t in range(a.every, len(paths), a.every):
            x = enc.preprocess([frame(i) for i in clip_indices(t)])
            real = enc.encode_tokens(x)                                                     # (2048, C)
            seen = np.nonzero(valid[: t + 1])[0]
            ref = int(seen[-1]) if len(seen) else 0                                         # last visible frame
            hide = enc.box_tokens(s.boxes[ref], w, h, last)
            pred_a = enc.predict(x, np.setdiff1d(np.arange(8 * g * g), hide), hide).mean(0)
            real_a = real[hide].mean(0)
            row = dict(seq=seq, t=t, absent=bool(s.absent[t]), target_seen_recently=bool(t - ref <= a.recent),
                       frames_since_seen=t - ref,
                       surprise=1 - cos(pred_a, real_a), mem_drop=1 - cos(real_a, m), mem_drop_pred=1 - cos(pred_a, m))
            if valid[t]:
                gt_idx = enc.box_tokens(s.boxes[t], w, h, 0)
                pred_b = enc.predict(x, ctx_b, tgt_b)[-g * g:]                              # predicted last slot
                for name, tok in (("real", real[last_tok]), ("predicted", pred_b),
                                  ("last_seen", real[enc.slot_tokens(a.context_slots - 1)])):
                    row[f"hit_{name}"], row[f"pct_{name}"] = search(m, tok, gt_idx)
                row["gt_share"] = len(gt_idx) / (g * g)
            rows.append(row)
            for i in [i for i in frames if i < t - 32]:
                del frames[i]
        pd.DataFrame(rows).to_csv(a.out / f"{seq}.csv", index=False)
        shutil.rmtree(seq_dir, ignore_errors=True)
        el = time.time() - t0
        print(f"[{k}/{len(todo)}] {seq}: {len(rows)} frames | elapsed {el / 60:.0f} min, "
              f"eta {el / k * (len(todo) - k) / 60:.0f} min", flush=True)

    files = sorted(a.out.glob("*.csv"))
    if files:
        summary = summarize(pd.concat(pd.read_csv(f) for f in files))
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
