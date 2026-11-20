"""
check_sleepfm_overlap.py

Which of OUR recordings did the released SleepFM see, and where? The released split file (sleepfm-clinical/sleepfm/configs/dataset_split.json)
lists recordings per split ('pretrain', 'train', 'validation', 'test', 'temporal_test', 'external_validation') with names such as
'mros/visit1/mros-visit1-aa2930.hdf5'. Names are mapped to our (cohort, subject_id, visit) keys and each of our visit pairs is classified:

    seen in pretraining   : at least one visit is in 'pretrain' (SleepFM saw it without labels) or in 'train'
    held out by SleepFM   : every visit found is only in 'validation' / 'test' / 'external_validation'
    not listed            : no visit of the subject appears in any list (filtered out or never in the split)

Why it matters: for MrOS most pairs will be 'seen in pretraining', so MrOS results come with the caveat "the model had seen these recordings",
and only the pairs held out by SleepFM are a clean MrOS test. SHHS is expected only in 'external_validation' (2,000 of ~5,800), WSC nowhere.

    python scripts/check_sleepfm_overlap.py --split /data2/sleepfm-clinical/sleepfm/configs/dataset_split.json
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict

PATTERNS = [
    ("mros", re.compile(r"mros-(visit[12])-([a-z]{2}\d+)", re.I), lambda m: (m.group(2).lower(), m.group(1).lower())),
    ("shhs", re.compile(r"(shhs[12])-(\d+)", re.I), lambda m: (m.group(2), m.group(1).lower())),
]


def parse_name(path):
    for cohort, rx, ex in PATTERNS:
        m = rx.search(path)
        if m:
            sid, visit = ex(m)
            return cohort, sid, visit
    return None


def classify(pair_splits):
    """pair_splits: set of split names in which any visit of the subject appears."""
    if not pair_splits:
        return "not listed"
    if pair_splits & {"pretrain", "train"}:
        return "seen in pretraining/train"
    return "held out by SleepFM (" + "/".join(sorted(pair_splits)) + ")"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--manifest", default="manifests/visit_embedding_manifest.csv")
    args = ap.parse_args()

    split = json.load(open(args.split))
    where = defaultdict(set)                                           # (cohort, sid, visit) -> splits
    unparsed = Counter()
    for name, files in split.items():
        for f in files:
            key = parse_name(f)
            if key is None:
                unparsed[name] += 1
            else:
                where[key].add(name)
    print("split file sizes:", {k: len(v) for k, v in split.items()})
    print("files not belonging to MrOS / SHHS (other cohorts), per split:", dict(unparsed))

    rows = list(csv.DictReader(open(args.manifest)))
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[(r["cohort"], r["subject_id"])].append(r["visit"])
    for cohort in ("shhs", "mros", "wsc"):
        subs = {s: v for (c, s), v in by_subject.items() if c == cohort}
        rec_in = Counter()
        for s, visits in subs.items():
            for v in visits:
                for sp in where.get((cohort, s, v), ()):
                    rec_in[sp] += 1
        n_rec = sum(len(v) for v in subs.values())
        print(f"\n=== {cohort}: {len(subs)} subjects, {n_rec} recordings in our manifest ===")
        print("  recordings found per split list:", dict(rec_in) or "none")
        cls, both = Counter(), Counter()
        for s, visits in subs.items():
            sp_all = set().union(*[where.get((cohort, s, v), set()) for v in visits])
            cls[classify(sp_all)] += 1
            per_visit = [bool(where.get((cohort, s, v), set()) & {"pretrain", "train"}) for v in visits]
            both["all visits seen" if all(per_visit) else "some visits seen" if any(per_visit) else "no visit seen"] += 1
        for k, v in cls.most_common():
            print(f"  subjects {k:45s} {v:5d}  ({100 * v / len(subs):.1f}%)")
        print("  seen-in-pretraining status over a subject's visits:", dict(both))
        if cohort == "mros":
            split_mix = Counter()
            for s, visits in subs.items():
                names = tuple(sorted({sp for v in visits for sp in where.get((cohort, s, v), ())}))
                split_mix[names] += 1
            print("  split membership pattern of subjects (a subject should sit in ONE split):", dict(split_mix))


if __name__ == "__main__":
    main()
