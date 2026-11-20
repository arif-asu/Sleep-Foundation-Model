"""
cache_sleepfm_tokens.py

Cache the INPUT of SleepFM's temporal transformer (`transformer_encoder`) for every recording, so that LoRA / DoRA adapters placed only in
`transformer_encoder` can be trained without running the frozen CNN tokenizer and the spatial attention pooling again at every step.

SetTransformer.forward is   patch_embedding -> spatial_pooling -> positional_encoding -> layer_norm -> transformer_encoder -> (temporal_pooling)
and the cache holds the tensor after `layer_norm` (what transformer_encoder receives), one 5-minute chunk (60 five-second tokens) at a time:

    outputs/sleepfm_tokens/<cohort>_<subject>_<visit>.npz
        tokens  (n_chunks, 4, 60, 128) float16     groups in the order BAS, EKG, EMG, RESP (as in extract_sleepfm_embeddings.py)
        valid   (n_chunks * 10, 4) bool            usable 30-s windows per group (same definition as before)
        n_windows, subject_id, cohort, visit
Preprocessing is exactly that of extract_sleepfm_embeddings.py (128 Hz, per-recording z-score on good windows, bad windows zeroed,
5-minute chunks, channel masks). The last partial chunk (< 5 min) of a recording is dropped.

Frozen path used for caching: tokenizer (BatchNorm in eval mode), spatial pooling and layer norm, dropout 0. A trained model must keep
exactly these parts frozen; --selftest checks that transformer_encoder(cache) equals the model's own forward pass.

    python scripts/cache_sleepfm_tokens.py --sleepfm_dir ... --model_dir ... --selftest
    python scripts/cache_sleepfm_tokens.py --sleepfm_dir ... --model_dir ... --limit 3
    python scripts/cache_sleepfm_tokens.py --sleepfm_dir ... --model_dir ... --drop-modalities AIRFLOW   # symmetric modality set across visits

About 6 MB per recording, about 55 GB for all 8,852 main-cohort recordings.
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from extract_sleepfm_embeddings import GROUPS, TARGET_HZ, WIN_SEC, prep_modality

CHUNK_WIN = 10                     # 10 windows of 30 s = 5 minutes = 60 tokens
TOKENS_PER_WIN = 6
DIM = 128


def encode_prefix(model, x, mask):
    """SetTransformer.forward up to, not including, transformer_encoder.
    x (B, C, T) float, T = n_tokens * patch_size; mask (B, C) bool, True = ignore the channel. Returns (B, S, 128)."""
    x = model.patch_embedding(x)                                   # (B, C, S, E)
    b, c, s, e = x.shape
    x = x.permute(0, 2, 1, 3).reshape(b * s, c, e)                 # 'b c s e -> (b s) c e'
    m = mask.unsqueeze(1).expand(-1, s, -1).reshape(b * s, c)      # 'b t c -> (b t) c'
    x = model.spatial_pooling(x, m)
    x = x.view(b, s, e)
    x = model.positional_encoding(x)
    return model.layer_norm(x)


def cache_recording(z, prefix_fn, chunk_win=CHUNK_WIN, batch_chunks=16, drop=()):
    """z: loaded npz of one recording. prefix_fn(x (B,C,T) float32, mask (B,C) bool) -> (B, S, 128) numpy.
    Returns tokens (n_chunks, 4, S, 128) float16 and valid (n_chunks * chunk_win, 4) bool, or None if shorter than one chunk."""
    n_win = int(z["n_windows"])
    names = [str(n) for n in z["modality_names"]]
    mmask = np.asarray(z["modality_mask"]).astype(bool)
    good = {n: mmask[:, i].copy() for i, n in enumerate(names)}
    prep = {}
    for n in names:
        try:
            raw = None if n in drop else np.asarray(z[n])
        except KeyError:
            raw = None
        y = None if raw is None else prep_modality(raw, good[n], n_win)
        if y is None:
            y = np.zeros((n_win, WIN_SEC * TARGET_HZ), np.float32)
            good[n] = np.zeros(n_win, bool)
        prep[n] = y
    n_full = n_win // chunk_win
    if n_full == 0:
        return None
    spc = chunk_win * WIN_SEC * TARGET_HZ
    tokens = np.zeros((n_full, len(GROUPS), chunk_win * TOKENS_PER_WIN, DIM), np.float16)
    valid = np.zeros((n_win, len(GROUPS)), bool)
    for gi, (_, mods) in enumerate(GROUPS.items()):
        mods = [m for m in mods if m in prep]
        if not mods:
            continue
        c = len(mods)
        valid[:, gi] = np.any([good[m] for m in mods], axis=0)
        sig = np.stack([prep[m].reshape(-1) for m in mods])                                   # (C, n_win * 3840)
        gmat = np.stack([good[m] for m in mods]).astype(np.float32)
        full = sig[:, : n_full * spc].reshape(c, n_full, spc).transpose(1, 0, 2)               # (n_full, C, spc)
        frac = gmat[:, : n_full * chunk_win].reshape(c, n_full, chunk_win).mean(axis=2).T      # (n_full, C)
        cmask = frac < 0.5                                                                      # True = ignore channel in this chunk
        cmask[cmask.all(axis=1)] = False                                                        # all bad: do not mask (avoids NaN)
        cmask = np.ascontiguousarray(cmask)
        for i in range(0, n_full, batch_chunks):
            tokens[i:i + batch_chunks, gi] = prefix_fn(np.ascontiguousarray(full[i:i + batch_chunks]), cmask[i:i + batch_chunks])
    return tokens, valid[: n_full * chunk_win]


def make_prefix_fn(model, device):
    import torch

    @torch.no_grad()
    def fn(x_np, mask_np):
        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device, dtype=torch.float32)
        m = torch.from_numpy(np.ascontiguousarray(mask_np)).to(device, dtype=torch.bool)
        return encode_prefix(model, x, m).float().cpu().numpy()

    return fn


def selftest(model, device):
    import torch
    torch.manual_seed(0)
    x = torch.randn(2, 3, CHUNK_WIN * WIN_SEC * TARGET_HZ, device=device)
    mask = torch.tensor([[False, False, True], [False, False, False]], device=device)
    with torch.no_grad():
        full = model(x, mask)[1]
        via_cache = model.transformer_encoder(encode_prefix(model, x, mask))
    diff = float((full - via_cache).abs().max())
    print(f"selftest: max |model(x, mask) tokens - transformer_encoder(cache)| = {diff:.2e} -> {'OK' if diff < 1e-4 else 'MISMATCH'}")
    return diff < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleepfm_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--manifest", default="manifests/full_manifest.csv")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out_dir", default="outputs/sleepfm_tokens")
    ap.add_argument("--cohorts", nargs="+", default=None)
    ap.add_argument("--drop-modalities", nargs="+", default=[], help="treat these modalities as missing at every visit")
    ap.add_argument("--batch_chunks", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from inspect_sleepfm import load_sleepfm
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_sleepfm(args.sleepfm_dir, args.model_dir, dropout=0.0, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    ok = selftest(model, device)
    if args.selftest or not ok:
        return

    rows = [r for r in csv.DictReader(open(args.manifest)) if r.get("status") == "ok" and r.get("npz_path")
            and (not args.cohorts or r["cohort"] in args.cohorts)]
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    prefix_fn = make_prefix_fn(model, device)
    t0, done, skipped = time.time(), 0, 0
    for i, r in enumerate(rows, 1):
        stem = os.path.splitext(os.path.basename(r["npz_path"]))[0]
        out = os.path.join(args.out_dir, stem + ".npz")
        if os.path.exists(out):
            continue
        z = np.load(os.path.join(args.root, r["npz_path"]), allow_pickle=False)
        res = cache_recording(z, prefix_fn, batch_chunks=args.batch_chunks, drop=tuple(args.drop_modalities))
        if res is None:
            skipped += 1
            continue
        tokens, valid = res
        tmp = out[:-4] + ".tmp.npz"
        np.savez(tmp, tokens=tokens, valid=valid, n_windows=tokens.shape[0] * CHUNK_WIN,
                 subject_id=str(z["subject_id"]), cohort=str(z["cohort"]), visit=str(z["visit"]))
        os.replace(tmp, out)
        done += 1
        if done % 100 == 0 or i == len(rows):
            rate = (time.time() - t0) / done
            print(f"[{i}/{len(rows)}] {done} written, {rate:.2f} s/recording, ETA {rate * (len(rows) - i) / 60:.0f} min", flush=True)
    print(f"done: {done} written, {skipped} shorter than one chunk, dropped modalities: {args.drop_modalities}")


if __name__ == "__main__":
    main()
