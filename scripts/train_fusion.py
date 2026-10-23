"""Train the V-JEPA fusion module on samples from scripts/prepare_fusion_data.py.

OSTrack (backbone + box head) and V-JEPA stay frozen; only src/tracker/fusion.FusionV1 (~2.6M params)
learns, with OSTrack's own losses (focal on the centre map + 2 x GIoU + 5 x L1). Samples whose target
is outside the search crop are skipped (OSTrack's head has no "absent" output; that case is fusion_v2).
Validation is on held-out VIDEOS; epoch 0 is evaluated before any update, i.e. it is exactly OSTrack.

    python scripts/train_fusion.py --data cache/fusion_data/motion --checkpoint OSTrack_ep0300.pth.tar
    python scripts/train_fusion.py ... --epochs 1 --batch 4                 # quick CPU check

Writes <out>/fusion.pth (best validation loss), history.csv and config.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.tracker.fusion import FusedOSTrack, FusionV1, align_to_crop  # noqa: E402
from src.tracker.ostrack import DEFAULT_CONFIG, setup_ostrack  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LOSS_WEIGHTS = dict(focal=1.0, giou=2.0, l1=5.0)  # OSTrack: cfg.TRAIN.GIOU_WEIGHT / L1_WEIGHT, focal 1


def load_samples(data_dir: Path) -> dict[str, np.ndarray]:
    parts, keys = [], ("template", "template_box", "search", "search_box", "crop", "kind", "in_crop", "vmap")
    for f in sorted(data_dir.glob("*.npz")):
        d = np.load(f)
        n = len(d["kind"])
        part = {k: d[k] for k in keys}
        part["frame_wh"] = np.repeat(d["frame_wh"][None], n, 0)
        part["video"] = np.array([f.stem] * n)
        parts.append(part)
    if not parts:
        sys.exit(f"no samples in {data_dir}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    keep = out["in_crop"].astype(bool)
    print(f"{len(keep)} samples from {len(parts)} videos, {keep.sum()} with the target in the search crop")
    return {k: v[keep] for k, v in out.items()}


def to_images(x: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN.to(device)) / STD.to(device)


def batch_losses(model, cfg, d: dict, idx: np.ndarray, device, objectives) -> dict:
    from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
    from lib.utils.ce_utils import generate_mask_cond
    from lib.utils.heapmap_utils import generate_heatmap

    template, search = to_images(d["template"][idx], device), to_images(d["search"][idx], device)
    t_box = torch.from_numpy(d["template_box"][idx]).float().to(device)
    s_box = torch.from_numpy(d["search_box"][idx]).float().to(device)
    v = align_to_crop(torch.from_numpy(d["vmap"][idx]).to(device), torch.from_numpy(d["crop"][idx]).float().to(device),
                      torch.from_numpy(d["frame_wh"][idx]).float().to(device))
    mask = generate_mask_cond(cfg, len(idx), device, t_box)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(template, search, v, ce_template_mask=mask)
    pred = box_cxcywh_to_xyxy(out["pred_boxes"].float()).view(-1, 4)
    gt = box_xywh_to_xyxy(s_box).clamp(0.0, 1.0)
    giou, iou = objectives["giou"](pred, gt)
    heat = generate_heatmap(s_box[None], cfg.DATA.SEARCH.SIZE, cfg.MODEL.BACKBONE.STRIDE)[0].unsqueeze(1)
    losses = dict(focal=objectives["focal"](out["score_map"].float(), heat.to(device)), giou=giou,
                  l1=F.l1_loss(pred, gt))
    losses["total"] = sum(LOSS_WEIGHTS[k] * losses[k] for k in LOSS_WEIGHTS)
    losses["iou"] = iou.detach()
    return losses


@torch.no_grad()
def evaluate(model, cfg, d, idx, device, objectives, batch) -> dict:
    model.eval()
    tot, ious = [], []
    for b in range(0, len(idx), batch):
        l = batch_losses(model, cfg, d, idx[b:b + batch], device, objectives)
        tot.append(float(l["total"]) * len(idx[b:b + batch]))
        ious.append(l["iou"].cpu().numpy())
    iou = np.concatenate(ious)
    kind = d["kind"][idx]
    return dict(loss=sum(tot) / len(idx), iou=float(iou.mean()),
                iou_occlusion=float(iou[kind == 1].mean()) if (kind == 1).any() else np.nan,
                iou_normal=float(iou[kind == 0].mean()) if (kind == 0).any() else np.nan,
                success_occlusion=float((iou[kind == 1] > 0.5).mean()) if (kind == 1).any() else np.nan)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=REPO_ROOT / "cache" / "fusion_data" / "motion")
    p.add_argument("--checkpoint", type=Path, required=True, help="OSTrack weights")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "fusion_v1")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1, help="share of VIDEOS held out")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_ostrack(a.out / "ostrack_output")
    from lib.models.ostrack import build_ostrack
    from lib.test.parameter.ostrack import parameters
    from lib.utils.box_ops import giou_loss
    from lib.utils.focal_loss import FocalLoss

    cfg = parameters(a.config).cfg
    net = build_ostrack(cfg, training=False)
    net.load_state_dict(torch.load(a.checkpoint, map_location="cpu")["net"], strict=True)
    model = FusedOSTrack(net, FusionV1()).to(device)
    objectives = dict(focal=FocalLoss(), giou=giou_loss)

    d = load_samples(a.data)
    # hold out whole videos so validation never sees frames of a training video
    is_val = np.array([zlib.crc32(v.encode()) % 1000 < a.val_frac * 1000 for v in d["video"]])
    if is_val.all() or not is_val.any():  # tiny runs: make sure both sides get at least one video
        is_val = d["video"] == d["video"][0]
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"train {len(tr)} samples / val {len(va)} samples ({len(set(d['video'][va]))} videos) on {device}")

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * max(1, int(np.ceil(len(tr) / a.batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))

    history = [dict(epoch=0, train_loss=np.nan, **{f"val_{k}": v for k, v in
                                                   evaluate(model, cfg, d, va, device, objectives, a.batch).items()})]
    print("epoch 0 (= OSTrack):", {k: round(v, 4) for k, v in history[0].items() if k.startswith("val")}, flush=True)
    best = history[0]["val_loss"]
    torch.save(model.fusion.state_dict(), a.out / "fusion.pth")  # zero-init = baseline, in case nothing improves
    rng = np.random.default_rng(a.seed)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = rng.permutation(tr)
        for b in range(0, len(tr), a.batch):
            l = batch_losses(model, cfg, d, perm[b:b + a.batch], device, objectives)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(l["total"].detach()))
        val = evaluate(model, cfg, d, va, device, objectives, a.batch)
        history.append(dict(epoch=ep, train_loss=float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}))
        if val["loss"] < best:
            best = val["loss"]
            torch.save(model.fusion.state_dict(), a.out / "fusion.pth")
        pd.DataFrame(history).to_csv(a.out / "history.csv", index=False)
        print(f"epoch {ep}: train {np.mean(losses):.4f} | val {val['loss']:.4f} | IoU occ {val['iou_occlusion']:.3f} "
              f"normal {val['iou_normal']:.3f} | {time.time() - t0:.0f}s", flush=True)
    print(f"best val loss {best:.4f} -> {a.out / 'fusion.pth'}")


if __name__ == "__main__":
    main()
