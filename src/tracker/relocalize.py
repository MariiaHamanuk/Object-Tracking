"""Training-free V-JEPA re-localization for OSTrack.

Every cached frame the target memory (first appearance + EMA of confident frames) is matched against
the whole V-JEPA map. When OSTrack doubts and the best match lies outside its search window, the best
places are proposed as candidates for OSTrack to verify (OSTrackRunner.track(verify=True)).

Optional parts: an occlusion warning from the V-JEPA predictor's surprise that freezes the memory,
a memory guard, a drift check for confident OSTrack, a motion prior for ranking candidates, and extra
candidates from YOLO-World or from the predictor's future map.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.tracker.fusion import VJEPACache, search_crop
from src.vjepa.encoder import pool_box

RANKED = ("score", "surp", "local", "rise", "online")
SIGNALS = {"score": ("score",), "both": ("score", "surp"), "local": ("score", "local"),
           "rise": ("score", "local", "rise"), "online": ("score", "online")}


@dataclass
class RelocParams:
    confident_score: float = 0.6
    ema: float = 0.1
    score_ratio: float = 0.75      # doubt: score below this fraction of the running mean
    margin: float = 0.10           # best similarity - similarity at OSTrack's box
    search_factor: float = 4.0
    candidates: int = 1
    min_sep: int = 2               # tokens between candidates
    surprise_gate: bool = False
    surprise_ratio: float = 1.5
    surprise_window: int = 64
    propose: bool = True           # False: only keep the memory up to date (prior head)
    yolo_k: int = 3
    warning: str = "off"           # off | score | both | local | rise | online | learned
    warn_window: int = 200
    warn_q: float = 0.95
    warn_hold: int = 16
    warn_relax: bool = False
    warn_score_ratio: float = 0.9
    warn_margin: float = 0.05
    warn_weights: str | None = None
    future_k: int = 0
    future_ahead: int = 2
    future_only_warned: bool = True
    memory_guard: float = 0.0
    drift_check: bool = False
    drift_margin: float = 0.15
    motion: float = 0.0


class RunningRank:
    """Rank of a value among the last `n` values (0.5 until `warmup` values)."""

    def __init__(self, n: int = 200, warmup: int = 10):
        self.h, self.warmup = deque(maxlen=n), warmup

    def rank(self, x: float) -> float:
        return 0.5 if len(self.h) < self.warmup else float((np.asarray(self.h) < x).mean())

    def add(self, x: float):
        self.h.append(x)


class VJEPARelocalizer:
    def __init__(self, cache: VJEPACache, frame_w: int, frame_h: int, init_box, params: RelocParams = RelocParams(),
                 detector=None, online=None):
        self.cache, self.w, self.h, self.p = cache, frame_w, frame_h, params
        self.detector, self.online = detector, online
        self.needs_image = detector is not None or online is not None
        if online is not None:
            online.reset()
        self.cached = set(int(t) for t in cache.frames)
        self.m_first = F.normalize(self._pool(0, init_box), dim=0)
        self.m_recent = self.m_first.clone()
        self.last_box = np.asarray(init_box, dtype=float)
        self.score_mean = None
        needs_surprise = params.surprise_gate or params.warning in ("both", "local", "rise", "learned")
        if needs_surprise and cache.surprise is None:
            raise ValueError(f"surprise_gate / warning={params.warning!r} need a V-JEPA cache made with "
                             "cache_vjepa.py --surprise")
        if params.warning == "online" and online is None:
            raise ValueError("warning='online' needs an OnlinePredictor")
        self.weights = json.load(open(params.warn_weights)) if params.warning == "learned" else None
        self.surprise_mean, self.last_surprise = None, -10**9
        self.ranks = {k: RunningRank(params.warn_window) for k in RANKED + ("first", "gap")}
        self.track_c, self.track_t, self.velocity = None, 0, np.zeros(2)
        self.signals = RunningRank(params.warn_window, warmup=20)
        self.prev_local = None
        self.warned, self.last_trigger = False, -10**9
        self.log = []

    def _map(self, t: int) -> torch.Tensor:
        return torch.from_numpy(self.cache.at(t).astype(np.float32))

    def _pool(self, t, box) -> torch.Tensor:
        return pool_box(self._map(t), box, self.w, self.h)

    def _signal(self, f: dict) -> float:
        if self.p.warning == "learned":
            w = self.weights
            return float(w["bias"] + sum(c * f[k] for k, c in w["coef"].items()))
        return float(np.mean([f[f"r_{k}"] for k in SIGNALS[self.p.warning]]))

    def __call__(self, i: int, box, score, image=None) -> np.ndarray | None:
        if self.online is not None and image is not None:
            self.online.push(i, image)
        confident = i == 0 or (np.isfinite(score) and score >= self.p.confident_score)
        learn = confident and not self.warned
        if learn:
            self.last_box = np.asarray(box, dtype=float)
            if i > 0:
                self.score_mean = score if self.score_mean is None else 0.9 * self.score_mean + 0.1 * score
        if i == 0 or i not in self.cached:
            return None

        fmap = self._map(i)
        cur = F.normalize(pool_box(fmap, box, self.w, self.h), dim=0)
        s_first = float((cur * self.m_first).sum())
        r_first = self.ranks["first"].rank(s_first)
        self.ranks["first"].add(s_first)
        agrees = self.p.memory_guard <= 0 or r_first >= self.p.memory_guard
        if learn and agrees:
            self.m_recent = F.normalize((1 - self.p.ema) * self.m_recent + self.p.ema * cur, dim=0)
        if learn and agrees and self.p.motion > 0:
            c = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
            if self.track_c is not None and i > self.track_t:
                self.velocity = 0.7 * self.velocity + 0.3 * (c - self.track_c) / (i - self.track_t)
            self.track_c, self.track_t = c, i
        m = F.normalize(self.m_first + self.m_recent, dim=0)
        sim = F.cosine_similarity(fmap, m[None, None], dim=-1)
        s_box = float(F.cosine_similarity(pool_box(fmap, box, self.w, self.h), m, dim=0))
        g = sim.shape[0]
        x1, y1, side = search_crop(box, self.p.search_factor)

        def centre(k):
            r, c = divmod(int(k), g)
            return (c + 0.5) * self.w / g, (r + 0.5) * self.h / g

        def is_outside(cx, cy):
            return not (x1 <= cx <= x1 + side and y1 <= cy <= y1 + side)

        order = torch.argsort(sim.flatten(), descending=True)
        s_best = float(sim.flatten()[order[0]])
        outside = is_outside(*centre(order[0]))

        raw = dict(score=-float(score))
        row = dict(frame=i, score=float(score), s_box=s_box, s_best=s_best, gap=s_best - s_box)
        if self.cache.surprise is not None:
            smap = self.cache.surprise_at(i).astype(np.float32)
            s_surp = float(pool_box(torch.from_numpy(smap)[..., None], self.last_box, self.w, self.h))
            surprised = self.surprise_mean is not None and s_surp > self.p.surprise_ratio * self.surprise_mean
            if surprised:
                self.last_surprise = i
            if learn and not surprised:
                self.surprise_mean = s_surp if self.surprise_mean is None else 0.9 * self.surprise_mean + 0.1 * s_surp
            local = s_surp - float(np.median(smap))
            raw.update(surp=s_surp, local=local, rise=0.0 if self.prev_local is None else local - self.prev_local)
            self.prev_local = local
            row.update(surprise=s_surp, surprised=bool(surprised), local=local, rise=raw["rise"])
        if self.online is not None:
            raw["online"] = self.online.target_surprise(i, self.last_box, self.w, self.h, fmap)
            row["online"] = raw["online"]
        # ranks over all previous checks; using only "normal" checks made the warning fire constantly
        for k in RANKED:
            row[f"r_{k}"] = self.ranks[k].rank(raw[k]) if k in raw else 0.5
            if k in raw:
                self.ranks[k].add(raw[k])

        if self.p.warning != "off":
            signal = self._signal(row)
            if len(self.signals.h) >= self.signals.warmup and signal > np.quantile(self.signals.h, self.p.warn_q):
                self.last_trigger = i
            self.signals.add(signal)
            self.warned = i - self.last_trigger <= self.p.warn_hold
            row.update(signal=signal, warned=self.warned)

        relaxed = self.warned and self.p.warn_relax
        ratio, margin = ((self.p.warn_score_ratio, self.p.warn_margin) if relaxed
                         else (self.p.score_ratio, self.p.margin))
        doubt = self.score_mean is not None and score < ratio * self.score_mean
        if self.p.surprise_gate:
            doubt = doubt and i - self.last_surprise <= self.p.surprise_window
        gap = s_best - s_box
        r_gap = self.ranks["gap"].rank(gap)
        self.ranks["gap"].add(gap)
        move = self.p.propose and doubt and outside and gap > margin
        drift = (self.p.propose and self.p.drift_check and not move and confident and outside
                 and gap > self.p.drift_margin and r_gap >= 0.95)
        row.update(doubt=bool(doubt), outside=outside, moved=bool(move or drift), drift=bool(drift),
                   r_first=r_first, memory_updated=bool(learn and agrees))
        self.log.append(row)
        if not (move or drift):
            return None

        n_pool = 3 * self.p.candidates if self.p.motion > 0 else self.p.candidates
        picked = []
        for k in order.tolist():
            r, c = divmod(k, g)
            if not is_outside(*centre(k)) or any(max(abs(r - pr), abs(c - pc)) < self.p.min_sep for pr, pc in picked):
                continue
            picked.append((r, c))
            if len(picked) == n_pool:
                break
        if self.p.motion > 0 and self.track_c is not None:
            pred = self.track_c + self.velocity * (i - self.track_t)
            diag = np.hypot(self.w, self.h)
            flat = sim.flatten()
            key = [float(flat[r * g + c]) - self.p.motion * np.hypot(*(np.array(centre(r * g + c)) - pred)) / diag
                   for r, c in picked]
            picked = [picked[j] for j in np.argsort(key)[::-1]]
        picked = picked[: self.p.candidates]
        w, h = self.last_box[2:]
        boxes = [np.array([cx - w / 2, cy - h / 2, w, h]) for cx, cy in (centre(r * g + c) for r, c in picked)]
        n_yolo = n_future = 0
        if self.detector is not None and image is not None:
            dets = [d for d in self.detector(image) if d[2] > 1 and d[3] > 1 and is_outside(d[0] + d[2] / 2, d[1] + d[3] / 2)]
            sims = [float(F.cosine_similarity(pool_box(fmap, d, self.w, self.h), m, dim=0)) for d in dets]
            for k in np.argsort(sims)[::-1][: self.p.yolo_k]:
                boxes.append(np.asarray(dets[k], dtype=float))
                n_yolo += 1
        if self.online is not None and self.p.future_k > 0 and (self.warned or not self.p.future_only_warned):
            fut = self.online.future_map(i, self.p.future_ahead)
            fsim = F.cosine_similarity(fut, m[None, None], dim=-1).flatten()
            taken = list(picked)
            for k in torch.argsort(fsim, descending=True).tolist():
                r, c = divmod(k, g)
                if not is_outside(*centre(k)) or any(max(abs(r - pr), abs(c - pc)) < self.p.min_sep for pr, pc in taken):
                    continue
                taken.append((r, c))
                cx, cy = centre(k)
                boxes.append(np.array([cx - w / 2, cy - h / 2, w, h]))
                n_future += 1
                if n_future == self.p.future_k:
                    break
        self.log[-1].update(n_vjepa=len(picked), n_yolo=n_yolo, n_future=n_future)
        return boxes[0] if self.p.candidates == 1 and n_yolo + n_future == 0 else np.stack(boxes)
