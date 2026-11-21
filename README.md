# SleepFM sleep-stage head: LSTM training code

Code that trains and tests the sleep-stage head on frozen SleepFM embeddings, one cohort at a time (SHHS, MrOS, WSC; HMC and Sleep-EDF as unseen cohorts).

Inputs it expects (not included): a token cache from `scripts/cache_sleepfm_tokens.py` (one 128-number token per 5 s and signal group, from SleepFM's encoder) and 30-s stage labels from `scripts/make_stage_labels.py`.

    python scripts/train_lstm_stage.py --datasets shhs mros wsc hmc sleepedf_sc --groups BAS --pooled-to-external

What it does: subject-wise 70/30 split per cohort (seed 42; both visits of a person on the same side); trains a 2-layer bidirectional LSTM on the 5-s tokens of visit 1 of the training subjects (10% of them pick the best epoch); tests on the held-out subjects' visit 1 and visit 2. HMC and Sleep-EDF: 5-fold by subject, or trained on the three main cohorts. Each 30-s label is copied to its six 5-s tokens; the six predictions are averaged for the 30-s epoch.

Reports balanced accuracy, macro-F1, recall per stage, bootstrap CIs, and the visit-2 minus visit-1 difference.

Files: `train_lstm_stage.py` and `sleepfm_probe_eval.py` (split and metric helpers) run the training; `cache_sleepfm_tokens.py` and `make_stage_labels.py` show how the inputs are made and need the rest of our pipeline (SleepFM loading helpers, the cohort registry and configs), which is not in this package.

Questions we want your view on: the split and validation scheme, how much training data is needed (SHHS: balanced accuracy 0.64 with 25 nights, 0.73 with 100, 0.79 with 400, 0.81 with all 1,663), and the 30-s to 5-s label handling.
