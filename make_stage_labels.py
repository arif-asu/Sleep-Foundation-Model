"""
make_stage_labels.py

Per-window sleep-stage labels aligned to the 30-s windows of the NPZ files (the 30-s task for the LoRA / DoRA adaptation).

Window w of a recording is epoch `onset_epoch + edge_sec/30 + w` of the annotation: the pipeline crops the NPZ signal to the sleep period
[onset_epoch, offset_epoch] and preprocess_modality trims edge_sec (60 s = 2 epochs) from the start. Labels: 0 W, 1 N1, 2 N2, 3 N3, 4 REM,
-1 = movement / unscored / no annotation (ignored in training).

Output: outputs/stage_labels/<cohort>_<subject>_<visit>.npy   int8 (n_windows,)

Alignment check (printed at the end): for a sample of recordings, the share of the variance of log(EEG 30-s std) that the stage labels
explain (R^2 of per-stage means, pooled over the sample) is computed after shifting the labels by -3 ... +3 epochs. EEG amplitude differs
strongly between stages (slow waves make N3 the largest), and windows at stage boundaries are the ones a one-epoch offset mislabels, so R^2
peaks where labels and signal are aligned. The peak must be at shift 0. If another shift is better by more than 1% (relative), the trim
assumption is wrong: tell me before training.

    python scripts/make_stage_labels.py --limit 20       # smoke test with the alignment table
    python scripts/make_stage_labels.py
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from src.data.annotations.base import Stage
from src.data.annotations.nsrr_xml import NSRRXMLParser
from src.data.annotations.wsc_gamma import WSCGammaParser
from src.data.annotations.wsc_twin import WSCTwinParser
from src.data.cohort_registry import CohortRegistry
from src.utils.edf_meta import read_edf_meta

PARSERS = {"nsrr_xml": NSRRXMLParser, "wsc_gamma": WSCGammaParser, "wsc_twin": WSCTwinParser}
CLASS = {Stage.WAKE: 0, Stage.N1: 1, Stage.N2: 2, Stage.N3: 3, Stage.REM: 4}
NAMES = ["W", "N1", "N2", "N3", "REM"]
EEG_HZ, WIN_SEC = 125, 30


def labels_for(stage_by_epoch, onset, trim_epochs, n_windows, shift=0):
    lab = np.full(n_windows, -1, np.int8)
    for w in range(n_windows):
        lab[w] = CLASS.get(stage_by_epoch.get(onset + trim_epochs + w + shift), -1)
    return lab


def alignment_table(samples, trim_epochs, shifts=range(-3, 4)):
    """samples: list of (eeg_window_std (n,), stage_by_epoch, onset). Returns {shift: R^2 of log(EEG std) explained by the stage labels}.
    Only windows that are labelled at every shift are used, so the shifts are compared on the same windows."""
    logs, labs = [], {s: [] for s in shifts}
    for std, sbe, onset in samples:
        per_shift = {s: labels_for(sbe, onset, trim_epochs, len(std), shift=s) for s in shifts}
        keep = np.all([per_shift[s] >= 0 for s in shifts], axis=0) & (std > 0)
        logs.append(np.log(std[keep]))
        for s in shifts:
            labs[s].append(per_shift[s][keep])
    x = np.concatenate(logs)
    table = {}
    for s in shifts:
        lab = np.concatenate(labs[s])
        sse = sum(((x[lab == k] - x[lab == k].mean()) ** 2).sum() for k in range(5) if (lab == k).sum() > 1)
        table[s] = float(1 - sse / ((x - x.mean()) ** 2).sum()) if len(x) > 10 else float("nan")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/full_manifest.csv")
    ap.add_argument("--out-dir", default="outputs/stage_labels")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--cohorts", nargs="+", default=["shhs", "mros", "wsc"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N recordings (0 = all)")
    ap.add_argument("--n-align", type=int, default=60, help="recordings used for the alignment table")
    ap.add_argument("--trim-epochs", type=int, default=None, help="override edge_sec/30")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    trim_epochs = args.trim_epochs if args.trim_epochs is not None else int(round(cfg.get("edge_sec", 60) / WIN_SEC))
    print(f"trim at the start of the sleep period: {trim_epochs} epochs")

    reg = CohortRegistry(args.config)
    lookup = {(c, r.subject_id, r.visit): r for c in args.cohorts for r in reg.list_recordings(c)}
    rows = [r for r in csv.DictReader(open(args.manifest))
            if r.get("status") == "ok" and r.get("npz_path") and r["cohort"] in args.cohorts]
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    dist, unlabeled, missing, samples = defaultdict(Counter), defaultdict(int), [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        if i % 500 == 0:
            print(f"  {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        key = (r["cohort"], r["subject_id"], r["visit"])
        rec = lookup.get(key)
        if rec is None:
            missing.append(key)
            continue
        z = np.load(r["npz_path"])
        onset, n_windows = int(z["onset_epoch"]), int(z["n_windows"])
        meta = read_edf_meta(rec.edf_path)
        parsed = PARSERS[rec.annotation_format]().parse(rec.annotation_path, meta.start_time)
        stage_by_epoch = {e.epoch_index: e.stage for e in parsed.epochs}
        lab = labels_for(stage_by_epoch, onset, trim_epochs, n_windows)
        np.save(os.path.join(args.out_dir, f"{r['cohort']}_{r['subject_id']}_{r['visit']}.npy"), lab)
        dist[r["cohort"]].update(lab.tolist())
        unlabeled[r["cohort"]] += int((lab < 0).sum())
        if len(samples) < args.n_align and "EEG" in z.files:
            eeg = np.asarray(z["EEG"], dtype=np.float64)
            win = WIN_SEC * EEG_HZ
            n = min(n_windows, len(eeg) // win)
            samples.append((eeg[: n * win].reshape(n, win).std(axis=1), stage_by_epoch, onset))

    print(f"\nlabels written to {args.out_dir}; recordings without an annotation record: {len(missing)}")
    for c, cnt in dist.items():
        tot = sum(cnt.values())
        print(f"  {c:5s} windows {tot:8d}  " + "  ".join(f"{NAMES[k]} {100 * cnt[k] / tot:4.1f}%" for k in range(5))
              + f"  ignored {100 * unlabeled[c] / tot:.1f}%")
    table = alignment_table(samples, trim_epochs)
    best = max(table, key=lambda s: (table[s] if table[s] == table[s] else -1))
    print(f"\nalignment check on {len(samples)} recordings: R^2 of log(EEG window std) explained by the stage labels, labels shifted by k epochs")
    print("  shift (epochs): " + "  ".join(f"{s:+7d}" for s in table))
    print("  R^2:            " + "  ".join(f"{v:7.4f}" for v in table.values()))
    gain = (table[best] - table[0]) / table[0] if table[0] > 0 else 0.0
    if best == 0:
        print("  peak at shift +0 -> aligned (expected)")
    elif gain <= 0.01:
        print(f"  peak at shift {best:+d}, but only {100 * gain:.2f}% better than shift 0 -> not distinguishable from zero offset at this sample size")
    else:
        print(f"  peak at shift {best:+d}, {100 * gain:.1f}% better than shift 0 -> NOT ALIGNED: the labels are offset, do not train on them")


if __name__ == "__main__":
    main()
