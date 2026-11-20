"""
train_lstm_stage.py

Sleep staging the way SleepFM's own paper fine-tunes it: frozen SleepFM embeddings of 5-second windows go through a 2-layer LSTM head that reads the
night as one sequence, and every dataset is trained and tested on its own.

Input: the cached transformer outputs of scripts/cache_sleepfm_tokens.py (tokens (n_chunks, 4 groups, 60, 128) = one 128-number vector per 5 seconds and
per signal group) and the 30-second stage labels. A 30-second label is given to its six 5-second windows.
Head: input LayerNorm -> (attention pooling over the available signal groups when several are chosen: a missing group is simply left out, as in
SleepFM's channel-agnostic design) -> 2-layer LSTM (bidirectional unless --unidirectional) -> linear layer with 5 outputs.
Training: random crops of --crop windows of each night, weighted cross-entropy, AdamW, best epoch picked on held-out training subjects.
Scoring: on whole nights, as 30-second epochs (probabilities of the six windows averaged; balanced accuracy and macro-F1, comparable with the other tables)
and as 5-second windows (macro-F1, comparable with the SleepFM paper).

Protocol per dataset (the dataset is never mixed with another one)
    SHHS, MrOS, WSC : the same 70/30 subject split as scripts/sleepfm_probe_eval.py (seed 42). Training: visit 1 of the training subjects. Test: visit 1 and
                      visit 2 of the held-out subjects (paired difference as before).
    HMC, Sleep-EDF  : subject-wise k-fold cross-validation inside the dataset (--ext-folds, default 5; both nights of a person always on the same side);
                      the scores are pooled over the test folds.
    --pooled-to-external additionally trains ONE model on the pooled training subjects of SHHS, MrOS and WSC and tests it on HMC and Sleep-EDF (the
                      'never seen' setting of the paper).
Notes
    * The token cache of the three cohorts was made with AIRFLOW dropped, so the RESP group there is THORAX + ABDOMEN + SpO2.
    * Tokens of different 5-minute chunks were produced independently by the transformer, so the sequence has small seams every 5 minutes.
    * --stub swaps the LSTM for a nearest-prototype classifier so that the whole data path can be tested without torch.

    python scripts/train_lstm_stage.py --datasets shhs --epochs 1 --limit-subjects 60            # smoke test
    python scripts/train_lstm_stage.py                                                          # all five datasets, BAS group
    python scripts/train_lstm_stage.py --groups BAS EKG EMG RESP --pooled-to-external           # all signal groups, SleepFM style
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from sleepfm_probe_eval import CLASSES, MAIN, VISIT1, VISIT2, boot_cm_ba, cm_metrics, read_csv, split_subjects

GROUPS = ["BAS", "EKG", "EMG", "RESP"]
TOK_PER_EPOCH = 6
PRINT_CM = False
CMS_STORE = {}                                                              # per-recording 30-s confusion matrices of every set, saved with --save-cms


# ================================================================================ numpy part (data path, splits, scoring)
def load_night(cache_file, label_file, gidx):
    """tokens (T, G, 128) float16 (T = n_chunks * 60), token mask (T, G) bool, 5-second labels (T,) int64 (-1 = ignore)."""
    z = np.load(cache_file)
    tok, valid = z["tokens"][:, gidx], z["valid"][:, gidx]                  # (n_chunks, G, 60, 128), (n_chunks * 10, G)
    if tok.ndim == 3:                                                       # dense cache of official_pipeline_subset.py --stride-sec: (T, G, 128) already in time order
        per = len(tok) // max(len(valid), 1)
        if per != TOK_PER_EPOCH:
            raise SystemExit(f"this cache holds {per} tokens per 30-s epoch: use --tokens-per-epoch {per}")
        n_ep = len(tok) // TOK_PER_EPOCH
        tok, valid = tok[: n_ep * TOK_PER_EPOCH], valid[:n_ep]
    else:                                                                    # chunked cache (n_chunks, G, 60, 128): consecutive 5-s windows, 6 per 30-s epoch
        if TOK_PER_EPOCH != 6:
            raise SystemExit("this cache holds consecutive 5-s windows (6 per 30-s epoch): use --tokens-per-epoch 6")
        n_chunks = tok.shape[0]
        tok = np.ascontiguousarray(tok.transpose(0, 2, 1, 3)).reshape(n_chunks * 60, len(gidx), 128)
        n_ep = n_chunks * 10
    lab = np.load(label_file).astype(np.int64)
    lab = np.concatenate([lab, -np.ones(max(0, n_ep - len(lab)), np.int64)])[:n_ep]
    lab = np.where(valid.any(axis=1), lab, -1)                              # a window in which none of the chosen groups is usable is not scored
    return tok, np.repeat(valid, TOK_PER_EPOCH, axis=0), np.repeat(lab, TOK_PER_EPOCH)


def night_paths(key, rows, cache_dir, label_dir):
    c, s, v = key
    r = rows.get(key)
    if r is None:
        return None
    cf, lf = os.path.join(cache_dir, os.path.basename(r["npz_path"])), os.path.join(label_dir, f"{c}_{s}_{v}.npy")
    return (cf, lf) if os.path.exists(cf) and os.path.exists(lf) else None


def preload(keys, rows, cache_dir, label_dir, gidx, max_nights=0, seed=0, tag=""):
    keys = list(keys)
    if max_nights and len(keys) > max_nights:
        keys = [keys[i] for i in np.random.default_rng(seed).permutation(len(keys))[:max_nights]]
    out, t0 = {}, time.time()
    for i, k in enumerate(keys, 1):
        p = night_paths(k, rows, cache_dir, label_dir)
        if p is not None:
            out[k] = load_night(*p, gidx)
        if i % 500 == 0:
            print(f"    {tag}loaded {i}/{len(keys)} nights ({time.time() - t0:.0f}s)", flush=True)
    return out


def epochs_from_tokens(probs):
    """(T, 5) token probabilities -> (n_epochs, 5): mean over the six windows of each 30-s epoch."""
    n = len(probs) // TOK_PER_EPOCH
    return probs[: n * TOK_PER_EPOCH].reshape(n, TOK_PER_EPOCH, -1).mean(axis=1)


def confusion(y, pred):
    m = y >= 0
    return np.bincount(y[m] * 5 + pred[m], minlength=25).reshape(5, 5)


def evaluate_stream(predict, keys, rows, cache_dir, label_dir, gidx):
    """Whole nights, one at a time. Returns {key: (30-s confusion matrix, 5-s confusion matrix)}."""
    out = {}
    for k in keys:
        p = night_paths(k, rows, cache_dir, label_dir)
        if p is None:
            continue
        tok, mask, tok_lab = load_night(*p, gidx)
        if (tok_lab >= 0).sum() == 0:
            continue
        probs = predict(tok, mask)
        n = len(probs) // TOK_PER_EPOCH
        ep_pred = epochs_from_tokens(probs).argmax(axis=1)
        out[k] = (confusion(tok_lab[: n * TOK_PER_EPOCH][::TOK_PER_EPOCH], ep_pred), confusion(tok_lab[: n * TOK_PER_EPOCH], probs[: n * TOK_PER_EPOCH].argmax(axis=1)))
    return out


def kfold_subjects(subjects, k, seed):
    subs = sorted(subjects)
    np.random.default_rng(seed).shuffle(subs)
    folds = [subs[i::k] for i in range(k)]
    return [(sorted(set(subs) - set(f)), sorted(f)) for f in folds]


def class_weights(labels, cap=8.0):
    counts = np.array([(labels == k).sum() for k in range(5)], float)
    return np.minimum(counts.sum() / (5 * np.maximum(counts, 1)), cap).astype(np.float32)


def sample_crop(night, length, rng):
    """A random crop of `length` windows (multiple of 6) of one night; shorter nights are padded (padding is not scored)."""
    tok, mask, lab = night
    t = len(tok)
    if t <= length:
        pad = length - t
        return (np.pad(tok, ((0, pad), (0, 0), (0, 0))), np.pad(mask, ((0, pad), (0, 0))), np.pad(lab, (0, pad), constant_values=-1))
    start = int(rng.integers(0, (t - length) // TOK_PER_EPOCH + 1)) * TOK_PER_EPOCH
    return tok[start:start + length], mask[start:start + length], lab[start:start + length]


def by_visit(res, label):
    """{key: ...} of one dataset -> ordered sets: each of two visit/night values alone, then everything pooled (only the pooled set when there is one value)."""
    vals = sorted({k[2] for k in res})
    if len(vals) != 2:
        return {label: res}, False
    out = {f"{label}, {v}": {k: x for k, x in res.items() if k[2] == v} for v in vals}
    out[f"{label}, both pooled"] = res
    return out, True


def report(title, results, seed, rows_out, paired=False):
    """results: {set name: {key: (cm30, cm5)}} -> printed table (+ CSV rows)."""
    if PRINT_CM:
        for name, res in results.items():
            if res:
                cm = np.stack([v[0] for v in res.values()]).sum(axis=0)
                print(f"[{title}] pooled 30-s confusion matrix, {name} (rows = expert stage, columns = predicted, in % of the expert stage; order {' '.join(CLASSES)}):")
                for k in range(5):
                    print("   " + CLASSES[k].rjust(4) + " " + " ".join(f"{100 * cm[k, j] / max(cm[k].sum(), 1):5.1f}" for j in range(5)))
    print(f"\n[{title}]")
    print(f"{'set':32s}{'recs':>6s}{'epochs':>9s}{'bal.acc [95% CI]':>22s}{'F1 30s':>8s}{'F1 5s':>7s}{'kappa':>7s}   " + " ".join(f"{k:>5s}" for k in CLASSES))
    for name, res in results.items():
        if not res:
            continue
        for k, v in res.items():
            CMS_STORE[f"{name}|{k[1]}|{k[2]}"] = v[0]
        a30 = np.stack([v[0] for v in res.values()])
        a5 = np.stack([v[1] for v in res.values()])
        m, m5 = cm_metrics(a30.sum(axis=0)), cm_metrics(a5.sum(axis=0))
        lo, hi = boot_cm_ba(a30, seed=seed)
        print(f"{name:32s}{len(res):6d}{m['n']:9d}{m['ba']:11.3f} [{lo:.3f},{hi:.3f}]{m['f1']:8.3f}{m5['f1']:7.3f}{m['kappa']:7.3f}   " + " ".join(f"{v:5.2f}" for v in m["recall"]))
        rows_out.append(dict(experiment=title, set=name, n_recordings=len(res), n_epochs=m["n"], ba=m["ba"], ba_lo=lo, ba_hi=hi, f1_30s=m["f1"], f1_5s=m5["f1"], kappa=m["kappa"],
                             **{f"recall_{k}": v for k, v in zip(CLASSES, m["recall"])}))
    if paired:
        names = list(results)
        a, b = results[names[0]], results[names[1]]
        subj = sorted({k[1] for k in a} & {k[1] for k in b})
        if len(subj) >= 20:
            ka = {k[1]: k for k in a}
            kb = {k[1]: k for k in b}
            c1 = np.stack([a[ka[s]][0] for s in subj])
            c2 = np.stack([b[kb[s]][0] for s in subj])
            d = cm_metrics(c2.sum(axis=0))["ba"] - cm_metrics(c1.sum(axis=0))["ba"]
            rng = np.random.default_rng(seed)
            diffs = []
            for _ in range(300):
                i = rng.integers(0, len(subj), len(subj))
                diffs.append(cm_metrics(c2[i].sum(axis=0))["ba"] - cm_metrics(c1[i].sum(axis=0))["ba"])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"  visit 2 minus visit 1, same subjects (balanced accuracy, paired bootstrap over subjects): n={len(subj)}  {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            rows_out.append(dict(experiment=title, set="visit2 - visit1 (paired)", n_recordings=len(subj), ba=d, ba_lo=lo, ba_hi=hi))


# ================================================================================ the two trainers
def make_stub_trainer():
    """Nearest-prototype classifier on the first group's tokens: lets the whole data path run without torch."""
    def train(train_nights, val_nights, n_groups):
        sums, cnt = np.zeros((5, 128)), np.zeros(5)
        for tok, mask, lab in train_nights.values():
            x = tok[:, 0].astype(np.float64)
            for k in range(5):
                m = (lab == k) & mask[:, 0]
                if m.any():
                    sums[k] += x[m].sum(axis=0)
                    cnt[k] += m.sum()
        proto = sums / np.maximum(cnt, 1)[:, None]
        proto /= np.linalg.norm(proto, axis=1, keepdims=True) + 1e-9

        def predict(tok, mask):
            x = tok[:, 0].astype(np.float64)
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
            s = 8.0 * x @ proto.T
            e = np.exp(s - s.max(axis=1, keepdims=True))
            return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        return predict
    return train


def make_torch_trainer(args):
    import copy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Stager(nn.Module):
        def __init__(self, n_groups):
            super().__init__()
            self.norm = nn.LayerNorm(128)
            self.att = nn.Linear(128, 1) if n_groups > 1 else None
            self.lstm = nn.LSTM(128, args.hidden, args.layers, batch_first=True, bidirectional=not args.unidirectional,
                                dropout=args.dropout if args.layers > 1 else 0.0)
            self.drop = nn.Dropout(args.dropout)
            self.fc = nn.Linear(args.hidden * (1 if args.unidirectional else 2), 5)

        def forward(self, x, mask):                                         # x (B, T, G, 128), mask (B, T, G) True = this group is usable here
            x = self.norm(x)
            if self.att is not None:
                s = self.att(x).squeeze(-1).masked_fill(~mask, -1e4)
                w = torch.softmax(s, dim=-1) * mask.float()
                w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                x = (x * w.unsqueeze(-1)).sum(dim=2)
            else:
                x = x[:, :, 0]
            h, _ = self.lstm(x)
            return self.fc(self.drop(h))                                    # (B, T, 5)

    class OfficialStager(nn.Module):
        """SleepFM's own SleepEventLSTMClassifier (models/models.py): attention pooling over the signal groups, positional encoding, LayerNorm, ONE
        transformer layer, ONE bidirectional LSTM layer (hidden 64 per direction), linear layer. Their forward reads the group mask at window 0 only;
        here it is applied window by window, and a window without any usable group keeps the first group so that the pooling never sees an empty set."""
        def __init__(self, n_groups):
            super().__init__()
            import importlib.util
            spec = importlib.util.spec_from_file_location("sleepfm_official_models", os.path.join(args.sleepfm_dir, "models", "models.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.net = mod.SleepEventLSTMClassifier(embed_dim=128, num_heads=4, num_layers=1, num_classes=5, pooling_head=4, dropout=args.dropout, max_seq_length=8196)

        def forward(self, x, mask):                                         # x (B, T, G, 128), mask (B, T, G) True = usable
            net = self.net
            b, t, g, e = x.shape
            usable = mask.clone()
            none = ~usable.any(dim=2)                                       # (B, T) no group usable: not scored, treated as padding in the transformer
            usable[:, :, 0] = usable[:, :, 0] | none
            none[:, 0] = none[:, 0] & ~none.all(dim=1)                      # never mask a whole sequence
            h = net.spatial_pooling(x.reshape(b * t, g, e), (~usable).reshape(b * t, g)).view(b, t, e)
            h = net.layer_norm(net.positional_encoding(h))
            h = net.transformer_encoder(h, src_key_padding_mask=none)
            h, _ = net.lstm(h)
            return net.fc(h)                                                # (B, T, 5)

    def make_predict(model):
        @torch.no_grad()
        def predict(tok, mask):
            model.eval()
            step = 8000 if args.head == "official" else max(len(tok), 1)    # the official head's positional encoding covers 8196 windows
            outs = []
            for a in range(0, len(tok), step):
                x = torch.from_numpy(tok[a:a + step].astype(np.float32)).unsqueeze(0).to(device)
                m = torch.from_numpy(np.ascontiguousarray(mask[a:a + step])).unsqueeze(0).to(device)
                outs.append(torch.softmax(model(x, m)[0], dim=-1).cpu().numpy())
            return np.concatenate(outs)
        return predict

    def val_ba(predict, nights):
        cm = np.zeros((5, 5), np.int64)
        for tok, mask, lab in nights.values():
            n = len(tok) // TOK_PER_EPOCH
            cm += confusion(lab[: n * TOK_PER_EPOCH][::TOK_PER_EPOCH], epochs_from_tokens(predict(tok, mask)).argmax(axis=1))
        return cm_metrics(cm)["ba"]

    def train(train_nights, val_nights, n_groups):
        if args.official_weights:
            model = OfficialStager(n_groups).to(device)
            try:
                ck = torch.load(args.official_weights, map_location="cpu", weights_only=False)
            except TypeError:
                ck = torch.load(args.official_weights, map_location="cpu")
            sd = ck
            if isinstance(ck, dict):
                for key in ("model_state_dict", "state_dict", "model"):
                    if isinstance(ck.get(key), dict):
                        sd = ck[key]
                        break
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            missing, unexpected = model.net.load_state_dict(sd, strict=False)
            print(f"    loaded SleepFM's own staging weights: {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
            if missing or unexpected:
                raise SystemExit(f"the weights do not fit the head: missing {missing[:3]}, unexpected {unexpected[:3]}")
            return make_predict(model)
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        model = (OfficialStager(n_groups) if args.head == "official" else Stager(n_groups)).to(device)
        print(f"    head parameters: {sum(p.numel() for p in model.parameters())}")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        weights = None if args.no_class_weights else torch.from_numpy(class_weights(np.concatenate([v[2] for v in train_nights.values()]))).to(device)
        keys = list(train_nights)
        length = max(TOK_PER_EPOCH, (args.crop // TOK_PER_EPOCH) * TOK_PER_EPOCH)
        best, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            crops = [sample_crop(train_nights[k], length, rng) for k in keys for _ in range(args.crops_per_night)]
            order, losses = rng.permutation(len(crops)), []
            for i in range(0, len(order), args.batch):
                b = [crops[j] for j in order[i:i + args.batch]]
                x = torch.from_numpy(np.stack([c[0] for c in b]).astype(np.float32)).to(device)
                m = torch.from_numpy(np.stack([c[1] for c in b])).to(device)
                y = torch.from_numpy(np.stack([c[2] for c in b])).to(device)
                if int((y >= 0).sum()) == 0:
                    continue
                loss = F.cross_entropy(model(x, m).reshape(-1, 5), y.reshape(-1), weight=weights, ignore_index=-1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            ba = val_ba(make_predict(model), val_nights) if val_nights else float("nan")
            flag = ""
            if not val_nights or ba > best + 1e-4:
                best, best_state, bad, flag = ba, copy.deepcopy(model.state_dict()), 0, "  <- best"
            else:
                bad += 1
            print(f"    epoch {epoch}/{args.epochs}  loss {np.mean(losses):.4f}  validation BA {ba:.4f}{flag}  ({time.time() - t0:.0f}s)", flush=True)
            if bad >= args.patience:
                print("    early stop")
                break
        model.load_state_dict(best_state)
        return make_predict(model)
    return train


# ================================================================================ experiments
def run_training(train_fn, train_keys, rows, args, gidx, tag):
    """Hold out a few training SUBJECTS for choosing the best epoch, preload the rest, train, return predict()."""
    if args.official_weights:
        return train_fn({}, {}, len(gidx))
    subs = sorted({k[1] for k in train_keys})
    rng = np.random.default_rng(args.seed + 1)
    rng.shuffle(subs)
    n_val = max(1, int(round(args.val_frac * len(subs)))) if len(subs) > 5 else 0
    val_subs = set(subs[:n_val])
    tr = [k for k in train_keys if k[1] not in val_subs]
    va = [k for k in train_keys if k[1] in val_subs][:150]
    t0 = time.time()
    train_nights = preload(tr, rows, args.cache_dir, args.label_dir_for(tr), gidx, args.max_train_nights, args.seed, f"{tag} train ")
    val_nights = preload(va, rows, args.cache_dir, args.label_dir_for(va), gidx, 0, args.seed, f"{tag} val ")
    if not train_nights:
        raise SystemExit(f"{tag}: no cached nights with labels found for training (check --cache-dir and the label folders)")
    print(f"  {tag}: training on {len(train_nights)} nights of {len(subs) - n_val} subjects, validation on {len(val_nights)} nights ({time.time() - t0:.0f}s to load)")
    return train_fn(train_nights, val_nights, len(gidx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["shhs", "mros", "wsc", "hmc", "sleepedf_sc"])
    ap.add_argument("--groups", nargs="+", default=["BAS"], choices=GROUPS)
    ap.add_argument("--pooled-to-external", action="store_true")
    ap.add_argument("--only-pooled", action="store_true", help="skip the per-dataset runs and do only the pooled -> unseen run (needs --pooled-to-external)")
    ap.add_argument("--train-visits", default="1", choices=["1", "2", "both"], help="main cohorts: which visit of the training subjects is used for training")
    ap.add_argument("--ext-folds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--unidirectional", action="store_true")
    ap.add_argument("--head", default="lstm", choices=["lstm", "official"],
                    help="lstm = our simple head; official = SleepFM's own SleepEventLSTMClassifier (group attention pooling, 1 transformer layer, 1-layer biLSTM with hidden 64)")
    ap.add_argument("--sleepfm_dir", default="/data2/sleepfm-clinical/sleepfm")
    ap.add_argument("--official-weights", default=None, help="path of SleepFM's released staging weights (checkpoints/model_sleep_staging/best.pth): NO training, the released stager is only evaluated")
    ap.add_argument("--print-confusion", action="store_true", help="also print the pooled 30-s confusion matrix (percent of each expert stage) of every set")
    ap.add_argument("--save-cms", default=None, help="save the 30-s confusion matrix of every recording (npz) for scripts/visit_effect_breakdown.py")
    ap.add_argument("--tokens-per-epoch", type=int, default=6, help="6 = consecutive 5-s windows; 10 = the dense cache of official_pipeline_subset.py --stride-sec 3 (5-s windows every 3 s); "
                    "with 10, also use --crop 2400 for a 2-hour training crop")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=1440, help="training crop length in 5-s windows (1440 = 2 hours)")
    ap.add_argument("--crops-per-night", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--max-train-nights", type=int, default=0, help="0 = all; limits memory when many groups are used")
    ap.add_argument("--limit-subjects", type=int, default=0, help="smoke test: first N subjects per dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--full-manifest", default="manifests/full_manifest.csv")
    ap.add_argument("--external-manifest", default="manifests/external_manifest.csv")
    ap.add_argument("--visit-manifest", default="manifests/visit_embedding_manifest.csv")
    ap.add_argument("--cache-dir", default="outputs/sleepfm_tokens")
    ap.add_argument("--label-dir", default="outputs/stage_labels")
    ap.add_argument("--ext-label-dir", default="outputs/stage_labels_external")
    ap.add_argument("--out", default="outputs/results/lstm_stage_eval.csv")
    args = ap.parse_args()
    global PRINT_CM
    PRINT_CM = args.print_confusion
    global TOK_PER_EPOCH
    TOK_PER_EPOCH = args.tokens_per_epoch
    if args.official_weights:
        args.head = "official"

    gidx = [GROUPS.index(g) for g in args.groups]
    main_rows = {(r["cohort"], r["subject_id"], r["visit"]): r for r in read_csv(args.full_manifest) if r.get("npz_path")}
    ext_rows = ({(r["cohort"], r["subject_id"], r["visit"]): r for r in read_csv(args.external_manifest) if r.get("status") == "ok" and r.get("npz_path")}
                if os.path.exists(args.external_manifest) else {})
    rows = {**main_rows, **ext_rows}
    args.label_dir_for = lambda keys: args.label_dir if keys and keys[0][0] in MAIN else args.ext_label_dir     # one label folder per call (keys of one dataset)

    by_cohort = defaultdict(set)                                            # exactly the construction of sleepfm_probe_eval.py, so the test subjects are identical
    for r in read_csv(args.visit_manifest):
        if r["cohort"] in MAIN:
            by_cohort[r["cohort"]].add(r["subject_id"])
    if args.limit_subjects:
        by_cohort = defaultdict(set, {c: set(sorted(s)[: args.limit_subjects]) for c, s in by_cohort.items()})
    test_subjects = split_subjects(by_cohort, args.test_frac, args.seed)
    train_fn = make_stub_trainer() if args.stub else make_torch_trainer(args)
    visits = {"1": [VISIT1], "2": [VISIT2], "both": [VISIT1, VISIT2]}[args.train_visits]
    direction = "uni" if args.unidirectional else "bi"
    head_desc = ("stub (nearest prototype)" if args.stub else "SleepFM's own SleepEventLSTMClassifier (transformer layer + biLSTM)" if args.head == "official"
                 else f"{args.layers}-layer {direction}directional LSTM, hidden {args.hidden}")
    print("groups " + "+".join(args.groups) + " | head: " + head_desc)
    out_rows = []
    clean_shhs = None
    if args.official_weights:                                  # the released stager was fine-tuned on SHHS, MESA, MrOS and SSC: only SHHS subjects of SleepFM's own test list are fair
        import json
        from check_sleepfm_overlap import parse_name
        split = json.load(open(os.path.join(args.sleepfm_dir, "configs", "dataset_split.json")))
        parsed = [parse_name(f) for f in split.get("external_validation", [])]
        clean_shhs = {p[1] for p in parsed if p and p[0] == "shhs"}
        print(f"SleepFM's released stager, evaluated as it is (no training here); SHHS is limited to the {len(clean_shhs)} subjects of its own test list")

    for ds in ([] if args.only_pooled else args.datasets):
        print(f"\n===== {ds} =====")
        if ds in MAIN:
            test = sorted(test_subjects[ds])
            train = sorted(by_cohort[ds] - test_subjects[ds])
            if args.official_weights:
                test = sorted(by_cohort[ds] & clean_shhs) if ds == "shhs" else sorted(by_cohort[ds])
                train = []
            predict = run_training(train_fn, [(ds, s, vm[ds]) for s in train for vm in visits], rows, args, gidx, ds)
            res = {"visit 1, held-out subjects": evaluate_stream(predict, [(ds, s, VISIT1[ds]) for s in test], rows, args.cache_dir, args.label_dir, gidx),
                   "visit 2, same subjects": evaluate_stream(predict, [(ds, s, VISIT2[ds]) for s in test], rows, args.cache_dir, args.label_dir, gidx)}
            report((f"{ds}: SleepFM's released stager, no training here" if args.official_weights else f"{ds}: trained on {ds} only"), res, args.seed, out_rows, paired=True)
        else:
            keys_by_subject = defaultdict(list)
            for k in ext_rows:
                if k[0] == ds:
                    keys_by_subject[k[1]].append(k)
            subjects = sorted(keys_by_subject)[: args.limit_subjects or None]
            if args.official_weights:
                predict = run_training(train_fn, [], rows, args, gidx, ds)
                pooled = evaluate_stream(predict, [k for s in subjects for k in keys_by_subject[s]], rows, args.cache_dir, args.ext_label_dir, gidx)
                sets, two = by_visit(pooled, "recordings")
                report(f"{ds}: SleepFM's released stager, no training here", sets, args.seed, out_rows, paired=two)
                continue
            pooled = {}
            for f, (tr_s, te_s) in enumerate(kfold_subjects(subjects, args.ext_folds, args.seed), 1):
                if not te_s or not tr_s:
                    continue
                print(f"  fold {f}/{args.ext_folds}: {len(tr_s)} training subjects, {len(te_s)} test subjects")
                predict = run_training(train_fn, [k for s in tr_s for k in keys_by_subject[s]], rows, args, gidx, f"{ds} fold {f}")
                pooled.update(evaluate_stream(predict, [k for s in te_s for k in keys_by_subject[s]], rows, args.cache_dir, args.ext_label_dir, gidx))
            sets, two = by_visit(pooled, "held-out recordings (pooled folds)")
            report(f"{ds}: trained on {ds} only ({args.ext_folds}-fold, subject-wise)", sets, args.seed, out_rows, paired=two)

    if (args.pooled_to_external or args.only_pooled) and not args.official_weights:
        print("\n===== pooled SHHS + MrOS + WSC -> unseen datasets =====")
        train_keys = [(c, s, vm[c]) for c in MAIN for s in sorted(by_cohort[c] - test_subjects[c]) for vm in visits]
        predict = run_training(train_fn, train_keys, rows, args, gidx, "pooled")
        res = {}
        for ds in args.datasets:
            if ds not in MAIN:
                sets, _ = by_visit(evaluate_stream(predict, sorted(k for k in ext_rows if k[0] == ds), rows, args.cache_dir, args.ext_label_dir, gidx), f"{ds} (never seen)")
                res.update(sets)
        report("pooled main cohorts -> unseen datasets", res, args.seed, out_rows)

    print("\nfor context, the SleepFM paper reports sleep-staging F1 of 0.70 to 0.78 across SSC, MESA, MrOS and SHHS and 0.55 on HMC without site-specific fine-tuning "
          "(2-layer LSTM on all four signal groups, 5-second windows; not the same protocol or test subjects).")
    if args.save_cms and CMS_STORE:
        os.makedirs(os.path.dirname(args.save_cms) or ".", exist_ok=True)
        np.savez(args.save_cms, **CMS_STORE)
        print(f"saved {len(CMS_STORE)} per-recording confusion matrices to {args.save_cms}")
    if out_rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        keys = sorted({k for r in out_rows for k in r})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(out_rows)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
