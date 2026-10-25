"""Train the fusion module, optionally the score head (--train-head) or only the prior head (--prior).

OSTrack's losses (focal + 2 GIoU + 5 L1); validation on held-out videos, epoch 0 = plain OSTrack.

    python scripts/train_fusion.py --data cache/fusion_data/motion --checkpoint OSTrack_ep0300.pth.tar
    python scripts/train_fusion.py ... --train-head score --negatives --out output/fusion_v1h
    python scripts/train_fusion.py ... --prior AB --negatives --lr 1e-3 --out output/fusion_v2a
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
from src.tracker.fusion import FusedOSTrack, FusionV1, PriorOSTrack, align_to_crop, prior_tokens  # noqa: E402
from src.tracker.ostrack import DEFAULT_CONFIG, setup_ostrack  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LOSS_WEIGHTS = dict(focal=1.0, giou=2.0, l1=5.0)  # as in OSTrack's config


def load_samples(data_dir: Path, negatives: bool) -> dict[str, np.ndarray]:
    parts, keys = [], ("template", "template_box", "search", "search_box", "crop", "kind", "in_crop", "vmap")
    for f in sorted(data_dir.glob("*.npz")):
        d = np.load(f)
        n = len(d["kind"])
        part = {k: d[k] for k in keys + tuple(k for k in ("mem_first", "mem_recent") if k in d.files)}
        part["frame_wh"] = np.repeat(d["frame_wh"][None], n, 0)
        part["video"] = np.array([f.stem] * n)
        parts.append(part)
    if not parts:
        sys.exit(f"no samples in {data_dir}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["in_crop"] = out["in_crop"].astype(bool)
    print(f"{len(out['kind'])} samples from {len(parts)} videos, {out['in_crop'].sum()} with the target in the "
          f"search crop{'' if negatives else ' (the others are skipped)'}")
    keep = np.ones(len(out["kind"]), bool) if negatives else out["in_crop"]
    return {k: v[keep] for k, v in out.items()}


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def to_images(x: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN.to(device)) / STD.to(device)


def batch_losses(model, cfg, d: dict, idx: np.ndarray, device, objectives, distill: float = 0.0) -> dict:
    """Positives (target in the crop): focal + GIoU + L1. Negatives: focal against an all-zero map.
    Distillation (training with a trainable head only): MSE to the original head's score map on
    visible normal frames."""
    from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
    from lib.utils.ce_utils import generate_mask_cond
    from lib.utils.heapmap_utils import generate_heatmap

    template, search = to_images(d["template"][idx], device), to_images(d["search"][idx], device)
    t_box = torch.from_numpy(d["template_box"][idx]).float().to(device)
    s_box = torch.from_numpy(d["search_box"][idx]).float().to(device)
    pos = torch.from_numpy(d["in_crop"][idx]).to(device)
    vmap, crop = torch.from_numpy(d["vmap"][idx]).to(device), torch.from_numpy(d["crop"][idx]).float().to(device)
    wh = torch.from_numpy(d["frame_wh"][idx]).float().to(device)
    if isinstance(model, PriorOSTrack):
        v = prior_tokens(vmap, torch.from_numpy(d["mem_first"][idx]).to(device),
                         torch.from_numpy(d["mem_recent"][idx]).to(device), crop, wh)
    else:
        v = align_to_crop(vmap, crop, wh)
    mask = generate_mask_cond(cfg, len(idx), device, t_box)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(template, search, v, ce_template_mask=mask)
    score_map = out["score_map"].float()
    heat = torch.zeros_like(score_map)
    if pos.any():
        heat[pos] = generate_heatmap(s_box[pos][None], cfg.DATA.SEARCH.SIZE,
                                     cfg.MODEL.BACKBONE.STRIDE)[0].unsqueeze(1).to(device)
    losses = dict(focal=objectives["focal"](score_map, heat))
    if pos.any():
        pred = box_cxcywh_to_xyxy(out["pred_boxes"].float()).view(-1, 4)[pos]
        gt = box_xywh_to_xyxy(s_box[pos]).clamp(0.0, 1.0)
        losses["giou"], iou = objectives["giou"](pred, gt)
        losses["l1"] = F.l1_loss(pred, gt)
    else:
        losses["giou"] = losses["l1"] = score_map.sum() * 0
        iou = torch.zeros(0, device=device)
    losses["total"] = sum(LOSS_WEIGHTS[k] * losses[k] for k in LOSS_WEIGHTS)
    normal = pos & torch.from_numpy(d["kind"][idx] == 0).to(device)
    if distill > 0 and "teacher_score_map" in out and normal.any():
        # score maps are mostly ~0, so the raw MSE is ~1e-3
        losses["distill"] = F.mse_loss(score_map[normal], out["teacher_score_map"].float()[normal]) * 100
        losses["total"] = losses["total"] + distill * losses["distill"]
    losses["iou"] = iou.detach()
    losses["peak"] = score_map.flatten(1).max(1).values.detach()
    return losses


@torch.no_grad()
def evaluate(model, cfg, d, idx, device, objectives, batch) -> dict:
    model.eval()
    tot, ious, peaks = [], [], []
    for b in range(0, len(idx), batch):
        l = batch_losses(model, cfg, d, idx[b:b + batch], device, objectives)
        tot.append(float(l["total"]) * len(idx[b:b + batch]))
        ious.append(l["iou"].cpu().numpy())
        peaks.append(l["peak"].cpu().numpy())
    iou, peak, pos = np.concatenate(ious), np.concatenate(peaks), d["in_crop"][idx]
    kind = d["kind"][idx][pos]
    return dict(loss=sum(tot) / len(idx), iou=float(iou.mean()),
                iou_occlusion=float(iou[kind == 1].mean()) if (kind == 1).any() else np.nan,
                iou_normal=float(iou[kind == 0].mean()) if (kind == 0).any() else np.nan,
                success_occlusion=float((iou[kind == 1] > 0.5).mean()) if (kind == 1).any() else np.nan,
                peak_pos=float(peak[pos].mean()), peak_neg=float(peak[~pos].mean()) if (~pos).any() else np.nan,
                peak_auroc=auroc(peak, pos))


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
    p.add_argument("--train-head", choices=["none", "score", "all"], default="none",
                   help="also fine-tune OSTrack's box head: its score branch or all of it")
    p.add_argument("--head-lr", type=float, default=1e-5)
    p.add_argument("--negatives", action="store_true", help="train on samples without the target in the crop")
    p.add_argument("--distill", type=float, default=1.0, help="weight of the distillation to OSTrack's head")
    p.add_argument("--prior", choices=["A", "AB"], default=None,
                   help="fusion_v2a: train only the V-JEPA memory prior in the score head (no feature fusion)")
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
    if a.prior:
        model = PriorOSTrack(net, a.prior).to(device)
    else:
        model = FusedOSTrack(net, FusionV1(), train_head=a.train_head).to(device)
    objectives = dict(focal=FocalLoss(), giou=giou_loss)

    d = load_samples(a.data, a.negatives)
    if a.prior and "mem_first" not in d:
        sys.exit("--prior needs samples with mem_first / mem_recent (re-run prepare_fusion_data.py)")
    # hold out whole videos
    is_val = np.array([zlib.crc32(v.encode()) % 1000 < a.val_frac * 1000 for v in d["video"]])
    if is_val.all() or not is_val.any():
        is_val = d["video"] == d["video"][0]
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"train {len(tr)} samples / val {len(va)} samples ({len(set(d['video'][va]))} videos) on {device}")

    if a.prior:
        head, groups = [], [dict(params=model.prior_head.new_parameters(), lr=a.lr)]
    else:
        head = [q for q in model.net.box_head.parameters() if q.requires_grad]
        groups = [dict(params=list(model.fusion.parameters()), lr=a.lr)] + ([dict(params=head, lr=a.head_lr)] if head else [])
    params = [q for g in groups for q in g["params"]]
    print(f"trainable: {sum(q.numel() for q in groups[0]['params'])} new parameters, "
          f"head {sum(q.numel() for q in head) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    steps = a.epochs * max(1, int(np.ceil(len(tr) / a.batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)

    def save():
        if a.prior:
            new = {n: q for n, q in model.prior_head.state_dict().items() if not n.startswith("head.")}
            torch.save(dict(mode=a.prior, prior_head=new), a.out / "prior.pth")
            return
        torch.save(model.fusion.state_dict(), a.out / "fusion.pth")
        if head:
            torch.save(model.net.box_head.state_dict(), a.out / "head.pth")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))

    history = [dict(epoch=0, train_loss=np.nan, **{f"val_{k}": v for k, v in
                                                   evaluate(model, cfg, d, va, device, objectives, a.batch).items()})]
    print("epoch 0 (= OSTrack):", {k: round(v, 4) for k, v in history[0].items() if k.startswith("val")}, flush=True)
    best = history[0]["val_loss"]
    save()
    rng = np.random.default_rng(a.seed)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = rng.permutation(tr)
        for b in range(0, len(tr), a.batch):
            l = batch_losses(model, cfg, d, perm[b:b + a.batch], device, objectives, a.distill)
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
            save()
        pd.DataFrame(history).to_csv(a.out / "history.csv", index=False)
        print(f"epoch {ep}: train {np.mean(losses):.4f} | val {val['loss']:.4f} | IoU occ {val['iou_occlusion']:.3f} "
              f"normal {val['iou_normal']:.3f} | peak pos {val['peak_pos']:.3f} neg {val['peak_neg']:.3f} "
              f"auroc {val['peak_auroc']:.3f} | {time.time() - t0:.0f}s", flush=True)
    print(f"best val loss {best:.4f} -> {a.out / 'fusion.pth'}")


if __name__ == "__main__":
    main()
