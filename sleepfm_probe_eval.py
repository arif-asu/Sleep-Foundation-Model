"""
sleepfm_probe_eval.py

Preliminary evaluation of the FROZEN pretrained SleepFM (no LoRA, no fine-tuning of the encoder): how well do its embeddings support two simple
supervised probes, and does that change from visit 1 to visit 2 and on data it has never seen?

  Probes are trained ONLY on visit 1 of TRAINING subjects (pooled over SHHS, MrOS, WSC; a cohort-stratified subject split, both visits of a subject on
  the same side). They are evaluated on
      (a) visit 1 of the held-out subjects        in-distribution reference
      (b) visit 2 of the SAME held-out subjects   a later recording of the same people (paired with (a))
      (c) external datasets (stage task only)     hmc, sleepedf_sc, ucddb: never used for training the probe, never seen by the probe

  stage task  per 30-s epoch, 5 classes (W, N1, N2, N3, REM), from the SleepFM BAS group (EEG + EOG, the only group every dataset has) with
              +- --context neighbouring epochs, multinomial logistic regression. Metrics: balanced accuracy, macro-F1, Cohen kappa, recall per
              class; 95% CI by bootstrap over recordings; visit-2 minus visit-1 by bootstrap over subjects (paired).
  osa task    per recording, SleepFM visit-level embedding (512 = BAS|EKG|EMG|RESP, columns chosen with --osa-cols) -> AHI severity
              (<5, 5-15, 15-30, >=30). Arms: embedding, age (age and age^2), embedding+age. Metrics: balanced accuracy (4 classes), AUROC for AHI >= 15.

What to keep in mind when reading it
  * SleepFM saw MrOS during pretraining (99% of our MrOS subjects, without labels); SHHS and WSC were not seen. MrOS rows carry that caveat.
  * Respiratory channels: SleepFM z-scores every recording, so amplitude differences do not matter, but a flat or dead airflow channel (SHHS visit 1)
    does. --osa-cols noresp / resp / all show how much the RESP group matters.
  * This is a baseline for Asif's question "how does SleepFM do on visit 2 and on unknown data", not a claim about longitudinal change.

Needs: outputs/sleepfm_emb (main cohorts), outputs/sleepfm_emb_external, labels from make_stage_labels.py and make_stage_labels_external.py,
outputs/visit_embeddings_sleepfm_compat.npz (osa).

    python scripts/sleepfm_probe_eval.py --tasks stage osa
    python scripts/sleepfm_probe_eval.py --tasks stage --limit-subjects 150     # smoke test
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

VISIT1 = {"shhs": "shhs1", "mros": "visit1", "wsc": "1"}
VISIT2 = {"shhs": "shhs2", "mros": "visit2", "wsc": "2"}
MAIN = ["shhs", "mros", "wsc"]
GROUPS = ["BAS", "EKG", "EMG", "RESP"]
CLASSES = ["W", "N1", "N2", "N3", "REM"]
RESULTS = []


# ----------------------------------------------------------------------------------------------- common helpers
def split_subjects(by_cohort, test_frac, seed):
    rng = np.random.default_rng(seed)
    test = {}
    for c, subs in by_cohort.items():
        subs = sorted(subs)
        rng.shuffle(subs)
        test[c] = set(subs[: max(1, int(len(subs) * test_frac))])
    return test


def balanced_accuracy(y, pred):
    """Mean recall over the classes that occur in y (no warning when a class is only predicted)."""
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))


def read_csv(path):
    return list(csv.DictReader(open(path)))


def cm_metrics(cm):
    cm = np.asarray(cm, float)
    rows, cols, diag = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    recall = np.where(rows > 0, diag / np.maximum(rows, 1), np.nan)
    prec = np.where(cols > 0, diag / np.maximum(cols, 1), np.nan)
    f1 = np.where((recall + prec) > 0, 2 * recall * prec / (recall + prec), np.nan)
    n = cm.sum()
    po = diag.sum() / n if n else np.nan
    pe = (rows * cols).sum() / n ** 2 if n else np.nan
    return {"ba": float(np.nanmean(recall)), "f1": float(np.nanmean(f1)), "kappa": float((po - pe) / (1 - pe)) if n and pe < 1 else np.nan,
            "recall": recall, "n": int(n)}


def boot_cm_ba(cms, n_boot=300, seed=0):
    """cms: (n_items, 5, 5) per-recording confusion matrices -> 95% CI of balanced accuracy, bootstrap over items."""
    rng = np.random.default_rng(seed)
    cms = np.asarray(cms, float)
    vals = []
    for _ in range(n_boot):
        vals.append(cm_metrics(cms[rng.integers(0, len(cms), len(cms))].sum(axis=0))["ba"])
    return np.percentile(vals, [2.5, 97.5])


# ----------------------------------------------------------------------------------------------- stage task
def load_stage_recording(emb_file, lab_file, gidx, context):
    z = np.load(emb_file)
    emb = z["emb"][:, gidx, :].astype(np.float32)
    valid = z["valid"][:, gidx].all(axis=1)
    lab = np.load(lab_file)
    n = min(len(emb), len(lab))
    emb, valid, lab = emb[:n], valid[:n], lab[:n]
    X = emb.reshape(n, -1)
    if context > 0:
        pad = np.pad(X, ((context, context), (0, 0)), mode="edge")
        X = np.concatenate([pad[i:i + n] for i in range(2 * context + 1)], axis=1)
    keep = valid & (lab >= 0)
    return X[keep], lab[keep].astype(int)


def stage_task(args, test_subjects, vrows):
    gidx = [GROUPS.index(g) for g in args.stage_groups]
    emb_dir, ext_emb_dir = args.emb_dir, args.ext_emb_dir
    main_rows = {(r["cohort"], r["subject_id"], r["visit"]): r for r in read_csv(args.full_manifest) if r.get("npz_path")}

    def paths(cohort, sid, visit, external=False):
        r = ext_rows[(cohort, sid, visit)] if external else main_rows[(cohort, sid, visit)]
        base = os.path.basename(r["npz_path"])
        return (os.path.join(ext_emb_dir if external else emb_dir, base),
                os.path.join(args.ext_label_dir if external else args.label_dir, f"{cohort}_{sid}_{visit}.npy"))

    ext_rows = {(r["cohort"], r["subject_id"], r["visit"]): r for r in read_csv(args.external_manifest)
                if r.get("status") == "ok" and r.get("npz_path")} if os.path.exists(args.external_manifest) else {}
    rng = np.random.default_rng(args.seed)
    by_subject = defaultdict(set)
    for r in vrows:
        by_subject[(r["cohort"], r["subject_id"])].add(r["visit"])
    subjects = {c: sorted(s for (cc, s), v in by_subject.items() if cc == c and {VISIT1[c], VISIT2[c]} <= v) for c in MAIN}
    if args.limit_subjects:
        subjects = {c: s[: args.limit_subjects] for c, s in subjects.items()}
    test = {c: {s for s in subjects[c] if s in test_subjects[c]} for c in MAIN}
    train = {c: [s for s in subjects[c] if s not in test_subjects[c]] for c in MAIN}

    Xs, ys, skipped = [], [], 0
    for c in MAIN:
        for s in train[c]:
            try:
                X, y = load_stage_recording(*paths(c, s, (VISIT1 if args.train_visit == 1 else VISIT2)[c]), gidx, args.context)
            except (FileNotFoundError, KeyError):
                skipped += 1
                continue
            if len(y) > args.max_epochs_per_rec:
                sel = rng.choice(len(y), args.max_epochs_per_rec, replace=False)
                X, y = X[sel], y[sel]
            Xs.append(X)
            ys.append(y)
    if not Xs:
        raise SystemExit("no training epochs found: check outputs/sleepfm_emb and outputs/stage_labels")
    X, y = np.vstack(Xs), np.concatenate(ys)
    print(f"\n[stage] probe on {'+'.join(args.stage_groups)} (context +-{args.context}): trained on {len(y)} epochs of "
          f"{sum(len(v) for v in train.values())} training subjects' visit {args.train_visit} ({skipped} recordings without embedding/labels skipped); "
          f"class balance " + " ".join(f"{CLASSES[k]} {100 * np.mean(y == k):.0f}%" for k in range(5)))
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=args.C, max_iter=500, class_weight="balanced").fit(scaler.transform(X), y)

    def evaluate(keys, external=False):
        out = {}
        for key in keys:
            try:
                X, y = load_stage_recording(*paths(*key, external=external), gidx, args.context)
            except (FileNotFoundError, KeyError):
                continue
            if len(y) == 0:
                continue
            pred = model.predict(scaler.transform(X))
            out[key] = np.bincount(y * 5 + pred, minlength=25).reshape(5, 5)
        return out

    results = {}
    for c in MAIN:
        results[(c, "visit 1, held-out subjects")] = evaluate([(c, s, VISIT1[c]) for s in sorted(test[c])])
        results[(c, "visit 2, same subjects")] = evaluate([(c, s, VISIT2[c]) for s in sorted(test[c])])
    for c in args.external_cohorts:
        keys = [k for k in ext_rows if k[0] == c]
        if keys:
            results[(c, "external, never seen")] = evaluate(sorted(keys), external=True)
            nights = sorted({k[2] for k in keys})
            if len(nights) == 2:                                   # e.g. Sleep-EDF SC: two nights per person -> also report each night alone
                for v in nights:
                    results[(c, f"external, {v}")] = evaluate(sorted(k for k in keys if k[2] == v), external=True)

    print(f"\n{'dataset':12s}{'set':28s}{'recs':>6s}{'epochs':>9s}{'bal.acc [95% CI]':>22s}{'F1':>7s}{'kappa':>7s}   " + " ".join(f"{k:>5s}" for k in CLASSES))
    for (c, name), cms in results.items():
        if not cms:
            continue
        arr = np.stack(list(cms.values()))
        m = cm_metrics(arr.sum(axis=0))
        lo, hi = boot_cm_ba(arr, seed=args.seed)
        note = "  (SleepFM saw MrOS in pretraining)" if c == "mros" else ""
        print(f"{c:12s}{name:28s}{len(cms):6d}{m['n']:9d}{m['ba']:11.3f} [{lo:.3f},{hi:.3f}]{m['f1']:7.3f}{m['kappa']:7.3f}   "
              + " ".join(f"{v:5.2f}" for v in m["recall"]) + note)
        RESULTS.append(dict(task="stage", dataset=c, set=name, n_recordings=len(cms), n_items=m["n"], ba=m["ba"], ba_lo=lo, ba_hi=hi,
                            f1=m["f1"], kappa=m["kappa"], **{f"recall_{k}": v for k, v in zip(CLASSES, m["recall"])}))
    print("\nvisit 2 minus visit 1 on the SAME held-out subjects (balanced accuracy, 95% CI by bootstrap over subjects; negative = worse at visit 2):")
    for c in MAIN:
        a, b = results.get((c, "visit 1, held-out subjects"), {}), results.get((c, "visit 2, same subjects"), {})
        subj = sorted({k[1] for k in a} & {k[1] for k in b})
        if len(subj) < 20:
            continue
        cm1 = np.stack([a[(c, s, VISIT1[c])] for s in subj])
        cm2 = np.stack([b[(c, s, VISIT2[c])] for s in subj])
        d = cm_metrics(cm2.sum(axis=0))["ba"] - cm_metrics(cm1.sum(axis=0))["ba"]
        brng, diffs = np.random.default_rng(args.seed), []
        for _ in range(300):
            i = brng.integers(0, len(subj), len(subj))
            diffs.append(cm_metrics(cm2[i].sum(axis=0))["ba"] - cm_metrics(cm1[i].sum(axis=0))["ba"])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  {c:5s} n={len(subj):4d}  {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        RESULTS.append(dict(task="stage", dataset=c, set="visit2 - visit1 (paired)", n_recordings=len(subj), ba=d, ba_lo=lo, ba_hi=hi))


# ----------------------------------------------------------------------------------------------- osa task
def osa_task(args, test_subjects, vrows):
    from src.data.clinical_linkage import load_all_ahi
    import check_visit_interval_confound as cvic

    ahi_by_cohort, age = load_all_ahi(), cvic.load_age_by_visit()
    comp = np.load(args.compat)[args.compat_key].astype(np.float64)
    cols = {"all": slice(0, 512), "noresp": slice(0, 384), "resp": slice(384, 512), "bas": slice(0, 128)}[args.osa_cols]
    rows = []
    for i, r in enumerate(vrows):
        c, s, v = r["cohort"], r["subject_id"], r["visit"]
        if c not in MAIN:
            continue
        a, ag = ahi_by_cohort.get(c, {}).get((s, v)), age.get((c, s, v))
        if a is not None and ag is not None:
            rows.append(dict(c=c, s=s, v=v, ahi=float(a), age=float(ag), i=i))
    have = defaultdict(set)
    for r in rows:
        have[(r["c"], r["s"])].add(r["v"])
    rows = [r for r in rows if {VISIT1[r["c"]], VISIT2[r["c"]]} <= have[(r["c"], r["s"])]]
    if args.limit_subjects:
        keep_subj = {c: set(sorted({r["s"] for r in rows if r["c"] == c})[: args.limit_subjects]) for c in MAIN}
        rows = [r for r in rows if r["s"] in keep_subj[r["c"]]]
    train = [r for r in rows if r["v"] == (VISIT1 if args.train_visit == 1 else VISIT2)[r["c"]] and r["s"] not in test_subjects[r["c"]]]
    test = [r for r in rows if r["s"] in test_subjects[r["c"]]]
    sev = lambda a: np.digitize(a, [5, 15, 30])
    feats = {"embedding": lambda rs: W_[[r["i"] for r in rs]],
             "age": lambda rs: np.array([[r["age"], r["age"] ** 2] for r in rs]),
             "embedding+age": lambda rs: np.hstack([W_[[r["i"] for r in rs]], np.array([[r["age"], r["age"] ** 2] for r in rs])])}
    W_ = comp[:, cols]
    print(f"\n[osa] SleepFM visit-level embedding columns '{args.osa_cols}' ({W_.shape[1]} dims); trained on {len(train)} visit-{args.train_visit} recordings of training subjects; "
          f"test {len({(r['c'], r['s']) for r in test})} held-out subjects, both visits")
    print(f"{'arm':16s}{'set':14s}{'cohort':>7s}{'n':>6s}{'bal.acc 4-class [95% CI]':>28s}{'AUROC AHI>=15 [95% CI]':>28s}")
    for arm, f in feats.items():
        scaler = StandardScaler().fit(f(train))
        model = LogisticRegression(C=args.osa_C, max_iter=3000, class_weight="balanced").fit(scaler.transform(f(train)), sev([r["ahi"] for r in train]))
        for label, vis in (("visit 1", VISIT1), ("visit 2", VISIT2)):
            for c in MAIN + ["pooled"]:
                rs = [r for r in test if r["v"] == (vis[r["c"]]) and (c == "pooled" or r["c"] == c)]
                if len(rs) < 30:
                    continue
                y = sev([r["ahi"] for r in rs])
                P = model.predict_proba(scaler.transform(f(rs)))
                pred = model.classes_[P.argmax(axis=1)]
                pos = P[:, [k for k, cl in enumerate(model.classes_) if cl >= 2]].sum(axis=1)
                ba = balanced_accuracy(y, pred)
                auc = roc_auc_score(y >= 2, pos) if len(set(y >= 2)) > 1 else np.nan
                brng, bs, au = np.random.default_rng(args.seed), [], []
                for _ in range(300):
                    j = brng.integers(0, len(rs), len(rs))
                    bs.append(balanced_accuracy(y[j], pred[j]))
                    if len(set(y[j] >= 2)) > 1:
                        au.append(roc_auc_score(y[j] >= 2, pos[j]))
                lo, hi, alo, ahi_ = *np.percentile(bs, [2.5, 97.5]), *np.percentile(au, [2.5, 97.5])
                note = "  (MrOS seen in SleepFM pretraining)" if c == "mros" else ""
                print(f"{arm:16s}{label:14s}{c:>7s}{len(rs):6d}{ba:12.3f} [{lo:.3f},{hi:.3f}]{auc:13.3f} [{alo:.3f},{ahi_:.3f}]{note}")
                RESULTS.append(dict(task=f"osa[{args.osa_cols}]", dataset=c, set=f"{arm} | {label}", n_recordings=len(rs), ba=ba, ba_lo=lo, ba_hi=hi,
                                    auroc=auc, auroc_lo=alo, auroc_hi=ahi_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["stage", "osa"], choices=["stage", "osa"])
    ap.add_argument("--full-manifest", default="manifests/full_manifest.csv")
    ap.add_argument("--external-manifest", default="manifests/external_manifest.csv")
    ap.add_argument("--visit-manifest", default="manifests/visit_embedding_manifest.csv")
    ap.add_argument("--emb-dir", default="outputs/sleepfm_emb")
    ap.add_argument("--ext-emb-dir", default="outputs/sleepfm_emb_external")
    ap.add_argument("--label-dir", default="outputs/stage_labels")
    ap.add_argument("--ext-label-dir", default="outputs/stage_labels_external")
    ap.add_argument("--compat", default="outputs/visit_embeddings_sleepfm_compat.npz")
    ap.add_argument("--compat-key", default="weighted")
    ap.add_argument("--train-visit", type=int, default=1, choices=[1, 2],
                    help="train the probes on visit 1 (default) or on visit 2 of the training subjects; evaluation sets stay the same")
    ap.add_argument("--stage-groups", nargs="+", default=["BAS"], choices=GROUPS)
    ap.add_argument("--external-cohorts", nargs="+", default=["hmc", "sleepedf_sc"],
                    help="external datasets for the stage task (ucddb is left out until its labels pass the alignment check)")
    ap.add_argument("--context", type=int, default=1, help="neighbouring epochs on each side appended to the features")
    ap.add_argument("--max-epochs-per-rec", type=int, default=60, help="training epochs sampled per recording")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--osa-cols", default="all", choices=["all", "noresp", "resp", "bas"])
    ap.add_argument("--osa-C", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-subjects", type=int, default=0, help="smoke test: only the first N subjects per cohort")
    ap.add_argument("--out", default="outputs/results/sleepfm_probe_eval.csv")
    args = ap.parse_args()

    vrows = read_csv(args.visit_manifest)
    by_cohort = defaultdict(set)
    for r in vrows:
        if r["cohort"] in MAIN:
            by_cohort[r["cohort"]].add(r["subject_id"])
    test_subjects = split_subjects(by_cohort, args.test_frac, args.seed)
    print("subject split (cohort-stratified; both visits of a subject on the same side): " +
          ", ".join(f"{c} test {len(test_subjects[c])} / train {len(by_cohort[c]) - len(test_subjects[c])}" for c in MAIN))
    if "stage" in args.tasks:
        stage_task(args, test_subjects, vrows)
    if "osa" in args.tasks:
        osa_task(args, test_subjects, vrows)
    if RESULTS:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        keys = sorted({k for r in RESULTS for k in r})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(RESULTS)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
