"""Run OSTrack on the LaSOT test set, streaming each video's frames from HuggingFace.

Only one or two videos are on disk at a time (the next one downloads while the current one is
tracked), so this runs on Kaggle / Colab without the 230 GB dataset. Finished videos are skipped,
so an interrupted run can simply be restarted.

    python scripts/run_ostrack_lasot.py --checkpoint OSTrack_ep0300.pth.tar
    python scripts/run_ostrack_lasot.py --checkpoint ... --shard 0 --num-shards 2   # one process per GPU
    python scripts/run_ostrack_lasot.py --checkpoint ... --sequences airplane-1 --max-frames 50  # smoke test

Writes <out>/<seq>.txt (x y w h per frame), <seq>_score.txt (confidence) and <seq>_time.txt.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.remote_zip import HF_FILE, retry, stream_sequences  # noqa: E402
from src.tracker.ostrack import DEFAULT_CONFIG, OSTrackRunner  # noqa: E402


def test_sequences() -> list[str]:
    text = retry(lambda: requests.get(HF_FILE.format("testing_set.txt"), timeout=60)).text
    return [s.strip() for s in text.splitlines() if s.strip()]


def load_box_file(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, default=None, help="default: results/ostrack/<config>/lasot")
    p.add_argument("--work", type=Path, default=REPO_ROOT / "tmp_frames", help="scratch dir for frames")
    p.add_argument("--sequences", default=None, help="comma-separated; default: official test set")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None, help="only for quick tests (output not comparable)")
    a = p.parse_args()

    out = a.out or REPO_ROOT / "results" / "ostrack" / a.config / "lasot"
    out.mkdir(parents=True, exist_ok=True)
    seqs = a.sequences.split(",") if a.sequences else test_sequences()
    seqs = seqs[a.shard::a.num_shards]
    todo = [s for s in seqs if not (out / f"{s}.txt").exists()]
    print(f"shard {a.shard}/{a.num_shards}: {len(seqs)} videos, {len(seqs) - len(todo)} done, {len(todo)} to run",
          flush=True)
    if not todo:
        return

    runner = OSTrackRunner(a.checkpoint, a.config, save_dir=a.work / "ostrack_output")
    t0, n_frames, failed = time.time(), 0, []
    for k, (seq, seq_dir, err) in enumerate(stream_sequences(todo, a.work / f"shard{a.shard}"), 1):
        if err is not None:
            failed.append(seq)
            print(f"[{k}/{len(todo)}] {seq}: download FAILED ({err})", flush=True)
            continue
        gt = load_box_file(seq_dir / "groundtruth.txt")
        frames = sorted((seq_dir / "img").glob("*.jpg"))[: a.max_frames]
        res = runner.track(frames, gt[0])
        np.savetxt(out / f"{seq}_score.txt", res["scores"], fmt="%.4f")
        np.savetxt(out / f"{seq}_time.txt", res["times"], fmt="%.5f")
        np.savetxt(out / f"{seq}.txt", res["boxes"], fmt="%.2f", delimiter="\t")  # written last = done marker
        shutil.rmtree(seq_dir, ignore_errors=True)
        n_frames += len(frames)
        el = time.time() - t0
        print(f"[{k}/{len(todo)}] {seq}: {len(frames)} frames, {len(frames) / res['times'].sum():.0f} fps model, "
              f"{n_frames / el:.0f} fps overall, elapsed {el / 60:.0f} min, "
              f"eta {el / k * (len(todo) - k) / 60:.0f} min", flush=True)
    if failed:
        sys.exit(f"failed: {','.join(failed)} - run again to retry")


if __name__ == "__main__":
    main()
