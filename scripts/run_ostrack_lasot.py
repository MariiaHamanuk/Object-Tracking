"""Run OSTrack on the LaSOT test set, streaming each video's frames from HuggingFace.

Only one or two videos are on disk at a time (the next one downloads while the current one is
tracked), so this runs on Kaggle / Colab without the 230 GB dataset. Finished videos are skipped,
so an interrupted run can simply be restarted.

    python scripts/run_ostrack_lasot.py --checkpoint OSTrack_ep0300.pth.tar
    python scripts/run_ostrack_lasot.py --checkpoint ... --shard 0 --num-shards 2   # one process per GPU
    python scripts/run_ostrack_lasot.py --checkpoint ... --sequences airplane-1 --max-frames 50  # smoke test
    python scripts/run_ostrack_lasot.py --checkpoint ... --sequences configs/splits/dev_occlusion.txt \\
        --vjepa-cache cache/vjepa --out results_raw/ostrack_vjepa_reloc   # + V-JEPA re-localization

Writes <out>/<seq>.txt (x y w h per frame), <seq>_score.txt (confidence) and <seq>_time.txt;
with --vjepa-cache also <seq>_reloc.csv (every V-JEPA check and whether the search region moved).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.lasot import read_split  # noqa: E402
from src.data.remote_zip import HF_FILE, retry, stream_sequences  # noqa: E402
from src.tracker.fusion import VJEPACache  # noqa: E402
from src.tracker.ostrack import DEFAULT_CONFIG, OSTrackRunner  # noqa: E402
from src.tracker.relocalize import RelocParams, VJEPARelocalizer  # noqa: E402


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
    p.add_argument("--sequences", default=None, help="split file or comma-separated names; default: test set")
    p.add_argument("--vjepa-cache", type=Path, default=None,
                   help="folder with <seq>.npz from scripts/cache_vjepa.py -> enable V-JEPA re-localization")
    p.add_argument("--reloc-verify", type=int, default=0, metavar="K",
                   help="re-localization proposes K places and OSTrack verifies them on the same frame "
                        "(0 = move to the best V-JEPA match without verification)")
    p.add_argument("--verify-min", type=float, default=0.0,
                   help="with --reloc-verify: OSTrack's score at the accepted candidate must also reach this")
    p.add_argument("--yolo", default=None, metavar="WEIGHTS",
                   help="with --reloc-verify: YOLO-World weights (e.g. yolov8s-worldv2.pt); its detections of the "
                        "video's class are added to the re-localization candidates (src/tracker/detector.py)")
    p.add_argument("--surprise-gate", action="store_true",
                   help="re-localize only after the V-JEPA predictor was surprised at the target "
                        "(cache from cache_vjepa.py --surprise)")
    p.add_argument("--fusion", type=Path, default=None,
                   help="fusion.pth from scripts/train_fusion.py: use the trained fusion (needs --vjepa-cache); "
                        "re-localization is then off unless --reloc-verify is given")
    p.add_argument("--prior", type=Path, default=None,
                   help="prior.pth from scripts/train_fusion.py --prior: OSTrack's head reads the V-JEPA target "
                        "memory prior (needs --vjepa-cache); re-localization only with --reloc-verify")
    p.add_argument("--fusion-head", type=Path, default=None,
                   help="head.pth from scripts/train_fusion.py --train-head (fine-tuned box head)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None, help="only for quick tests (output not comparable)")
    a = p.parse_args()

    out = a.out or REPO_ROOT / "results" / "ostrack" / a.config / "lasot"
    out.mkdir(parents=True, exist_ok=True)
    if a.sequences and Path(a.sequences).is_file():
        seqs = sorted(read_split(a.sequences))
    else:
        seqs = a.sequences.split(",") if a.sequences else test_sequences()
    if (a.fusion or a.prior) and not a.vjepa_cache:
        sys.exit("--fusion / --prior need --vjepa-cache")
    if a.fusion and a.prior:
        sys.exit("use either --fusion or --prior")
    if a.vjepa_cache:
        missing = [s for s in seqs if not (a.vjepa_cache / f"{s}.npz").exists()]
        if missing:
            sys.exit(f"no V-JEPA cache for {len(missing)} videos, e.g. {missing[:3]}")
    seqs = seqs[a.shard::a.num_shards]
    todo = [s for s in seqs if not (out / f"{s}.txt").exists()]
    print(f"shard {a.shard}/{a.num_shards}: {len(seqs)} videos, {len(seqs) - len(todo)} done, {len(todo)} to run",
          flush=True)
    if not todo:
        return

    runner = OSTrackRunner(a.checkpoint, a.config, save_dir=a.work / "ostrack_output")
    if a.fusion:
        runner.enable_fusion(a.fusion, a.fusion_head)
    if a.prior:
        runner.enable_prior(a.prior)
    detector = None
    if a.yolo:
        if not a.reloc_verify:
            sys.exit("--yolo needs --reloc-verify")
        from src.tracker.detector import YOLODetector
        detector = YOLODetector(a.yolo)
    t0, n_frames, failed = time.time(), 0, []
    for k, (seq, seq_dir, err) in enumerate(stream_sequences(todo, a.work / f"shard{a.shard}"), 1):
        if err is not None:
            failed.append(seq)
            print(f"[{k}/{len(todo)}] {seq}: download FAILED ({err})", flush=True)
            continue
        gt = load_box_file(seq_dir / "groundtruth.txt")
        frames = sorted((seq_dir / "img").glob("*.jpg"))[: a.max_frames]
        hook = None
        if a.vjepa_cache:
            h, w = cv2.imread(str(frames[0])).shape[:2]
            cache = VJEPACache(a.vjepa_cache / f"{seq}.npz")
            trained = a.fusion or a.prior
            if not trained or a.reloc_verify > 0 or a.prior:  # the prior needs the relocalizer's memory
                params = RelocParams(candidates=max(a.reloc_verify, 1), surprise_gate=a.surprise_gate,
                                     propose=not trained or a.reloc_verify > 0)
                if detector is not None:
                    detector.set_class(seq.rsplit("-", 1)[0])
                hook = VJEPARelocalizer(cache, w, h, gt[0], params, detector=detector)
            if trained:
                runner.set_vjepa(cache, w, h, memory=hook)
        res = runner.track(frames, gt[0], hook=hook, verify=a.reloc_verify > 0, verify_min=a.verify_min)
        if hook is not None:
            pd.DataFrame(hook.log).to_csv(out / f"{seq}_reloc.csv", index=False)
            if res["checks"]:
                pd.DataFrame(res["checks"]).to_csv(out / f"{seq}_verify.csv", index=False)
        np.savetxt(out / f"{seq}_score.txt", res["scores"], fmt="%.4f")
        np.savetxt(out / f"{seq}_time.txt", res["times"], fmt="%.5f")
        np.savetxt(out / f"{seq}.txt", res["boxes"], fmt="%.2f", delimiter="\t")  # written last = done marker
        shutil.rmtree(seq_dir, ignore_errors=True)
        n_frames += len(frames)
        el = time.time() - t0
        print(f"[{k}/{len(todo)}] {seq}: {len(frames)} frames, {len(frames) / res['times'].sum():.0f} fps model, "
              f"{n_frames / el:.0f} fps overall, moved {len(res['moved'])}x, elapsed {el / 60:.0f} min, "
              f"eta {el / k * (len(todo) - k) / 60:.0f} min", flush=True)
    if failed:
        sys.exit(f"failed: {','.join(failed)} - run again to retry")


if __name__ == "__main__":
    main()
