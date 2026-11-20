# Frozen SleepFM on longitudinal polysomnography: sleep-stage evaluation

Does SleepFM's pretrained encoder, and its sleep-stage head, hold up on a second night of the same people recorded years later, and on cohorts it did not see?
Cohorts: SHHS, MrOS and WSC (visit 1 and visit 2); HMC and Sleep-EDF (unseen by SleepFM).
Research code, work in progress. No data, embeddings or SleepFM files are included.

## Findings so far

Balanced accuracy (BA); V1 and V2 are visit 1 and visit 2. Numbers come from our runs; the project document has the details and the caveats.

| Setting | Result |
| --- | --- |
| SleepFM's own preprocessing and its released stage head, no training here | SHHS BA 0.82 (V1) and 0.84 (V2), 200 subjects of SleepFM's own test list; WSC 0.74 and 0.69, 150 subjects, a cohort SleepFM never saw |
| The same head on embeddings from our own preprocessing | BA 0.37 to 0.56: our preprocessing differs from SleepFM's |
| LSTM trained on frozen 5-second embeddings (EEG and EOG group, our preprocessing), per cohort, V1 and V2 | SHHS 0.81 and 0.81, MrOS 0.79 and 0.74, WSC 0.79 and 0.77 |
| Unseen cohorts, LSTM trained on SHHS, MrOS and WSC | HMC 0.66, Sleep-EDF 0.75 |
| V2 minus V1 | SHHS about 0; MrOS -0.04 to -0.05 with every method (cause unknown); WSC -0.02 to -0.05, most likely because experts labelled N3 more loosely at V2 (N3 share 3.7% to 8.9%) |
| 5-second windows with 2 seconds of overlap | no gain (3 subject splits, mean change within 0.008) |

## What is in the repository

| Path | Purpose |
| --- | --- |
| `scripts/build_dataset*.py`, `src/`, `configs/` | EDF files to our NPZ format (channel choice, filters, windows) and the cohort registry |
| `scripts/make_stage_labels*.py` | expert stage labels aligned to the NPZ windows |
| `scripts/cache_sleepfm_tokens.py` | frozen SleepFM encoder: one 128-number token per 5 seconds and signal group |
| `scripts/train_lstm_stage.py` | the sleep-stage head (our LSTM, SleepFM's own head, or SleepFM's released weights), train and test one cohort at a time |
| `scripts/sleepfm_probe_eval.py` | logistic-regression baseline, split and metric helpers used by the other scripts |
| `scripts/official_pipeline_subset.py` | SleepFM's own preprocessing and embedding recipe on a subset of recordings; `--stride-sec 3` gives 5-second windows with 2 seconds of overlap |
| `scripts/inspect_sleepfm.py`, `extract_sleepfm_embeddings.py`, `lora_sleepfm.py` | loading SleepFM and running its encoder |
| `scripts/check_*.py`, `channel_census.py`, `candidate_coverage.py`, `visit_effect_breakdown.py`, `n3_contrast_by_visit.py` | audits to run before trusting a visit-2 versus visit-1 comparison |

## Setup

- Python 3.12. Packages: see `requirements.txt` (made with `pip freeze` in the environment we used); SleepFM's preprocessing also needs `mne`, `pyedflib`, `h5py` and `matplotlib`.
- SleepFM: clone https://github.com/zou-group/sleepfm-clinical (we used commit 2bcbae0) and put its weights in `sleepfm/checkpoints/model_base` and `model_sleep_staging`. Pass the folders with `--sleepfm_dir` and `--model_dir`. Check its licence before redistributing anything.
- Data: SHHS, MrOS and WSC from the NSRR (data-use agreements apply); HMC and Sleep-EDF from PhysioNet. Set the paths in `configs/config.yaml`. Run every script from the repository root.

## Pipeline A: our own preprocessing

```bash
python scripts/build_dataset.py --help                       # EDF to NPZ and manifests (SHHS, MrOS, WSC); external cohorts: scripts/build_dataset_external.py
python scripts/make_stage_labels.py                          # labels for SHHS, MrOS, WSC
python scripts/make_stage_labels_external.py                 # labels for HMC, Sleep-EDF
python scripts/cache_sleepfm_tokens.py --sleepfm_dir <sleepfm>/sleepfm --model_dir <sleepfm>/sleepfm/checkpoints/model_base --drop-modalities AIRFLOW
python scripts/train_lstm_stage.py --datasets shhs mros wsc hmc sleepedf_sc --groups BAS --pooled-to-external
```

`--drop-modalities AIRFLOW` keeps the breathing group the same at both visits (SHHS visit 1 has no airflow channel we recognise in a third of its files). For the external cohorts the cache script takes the external manifest with `--manifest manifests/external_manifest.csv` (see `--help`).

## Pipeline B: SleepFM's own preprocessing

```bash
python scripts/official_pipeline_subset.py --sleepfm_dir <sleepfm>/sleepfm --model_dir <sleepfm>/sleepfm/checkpoints/model_base \
    --wsc 150 --shhs 200 --out-root outputs/official
python scripts/train_lstm_stage.py --official-weights <sleepfm>/sleepfm/checkpoints/model_sleep_staging/best.pth \
    --groups BAS EKG EMG RESP --cache-dir outputs/official/tokens --label-dir outputs/official/labels --datasets wsc shhs --print-confusion
```

SleepFM's code skips a file that lacks a channel for any of its four signal groups, so HMC (no breathing) and Sleep-EDF (no ECG) cannot go through it. Its channel list does not know WSC's channel names; `WSC_EXTRA` in the script adds them (our choice, not SleepFM's). SHHS is limited to SleepFM's own test list because the released stage head was fine-tuned on other SHHS subjects.

## Evaluation protocol

- Subject-wise 70/30 split per cohort (seed 42); both visits of a person are always on the same side.
- Train on visit 1 of the training subjects; test on the held-out subjects' visit 1 and visit 2 (the same people). 10% of the training subjects choose the best epoch.
- HMC and Sleep-EDF: 5-fold cross-validation by subject (both nights of a person together), or one model trained on the three main cohorts and tested on all of them.
- Metrics: balanced accuracy, macro-F1 (30-second epochs and 5-second windows), recall per stage, bootstrap confidence intervals over people.

## How 30-second labels meet 5-second tokens

SleepFM's stage head has no mapping layer between the two. Its encoder gives one token per 5 seconds (`patch_size 640` at 128 Hz), the head ends in `Linear(128, 5)` applied to every token, and the loss compares those outputs with the label rows one to one. We copy each 30-second label to its six tokens and average the six predictions to get the 30-second prediction. Whether SleepFM expands its labels to 5 seconds when it builds its label files is our reading of its configuration, not something we could check (the label files are not in its repository).

## Known limits and open questions

- Our preprocessing (band-pass, cropping to the sleep period, window masking, one to four channels per group) differs from SleepFM's (whole-night z-score, 64 Hz low-pass, every matching channel); SleepFM's released stage head does not work on our embeddings. Which of our steps causes the mismatch is not known.
- MrOS visit 2 is lower with every method and the cause is not found; 91.6% of its visit-2 recordings are in SleepFM's training lists, and 165 MrOS subjects are split across SleepFM's lists, which the SleepFM paper says it avoids.
- WSC stage labels do not mean the same at both visits (N3 criterion); its instrument also changes from visit 3.
- Results with SleepFM's own preprocessing are on subsets (WSC 150, SHHS 200 subjects); HMC and Sleep-EDF results come from our preprocessing.
- The 2-second-overlap test used 188 WSC and 248 SHHS training nights.

## Not included

No data, embeddings, manifests with subject IDs, SleepFM code or SleepFM weights. Licence for this code: not chosen yet.
