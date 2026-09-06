# Problem 01 — Validation experiment log

This log records the experiments actually performed for Problem 01
(NORMAL vs. PNEUMONIA classification on pediatric chest X-rays). Only
genuinely executed runs are listed; every figure is taken from the
artifacts under `results/` of the corresponding run.

## Run register

| # | Run ID | Description | Seed | Best epoch (val PR-AUC) | Val AP | Val recall @ threshold | Val F1 @ threshold | Selected? |
|---|--------|-------------|------|-------------------------|--------|------------------------|--------------------|-----------|
| 1 | `20260905T121522Z_f27df8` | Baseline from-scratch CNN (see Configuration) | 42 | 27 | 0.9992 | 0.9865 | 0.9865 | Yes |

The exact split fingerprints are recorded in
`results/dataset_audit.json` of the selected run.

## Configuration (run `20260905T121522Z_f27df8`)

- Model: from-scratch CNN with four convolutional blocks
  (filters 32 / 64 / 128 / 256), global average pooling, dense layer
  (128 units), dropout 0.40, L2 regularization 1e-4.
- Input: single-channel grayscale, 180 × 180 pixels; rescaling (1/255)
  applied as the first in-model layer.
- Optimizer: Adam, learning rate 1e-3; batch size 32.
- Maximum 30 epochs with early stopping (patience 6) on validation
  PR-AUC; training completed 30 epochs and the best checkpoint was from
  epoch 27.
- Class weights derived from the final training split to counter class
  imbalance.
- Decision rule: a single threshold was chosen on the validation set by
  maximizing the PNEUMONIA-class F1 (ties broken by the largest
  threshold) and then frozen in `inference_config.json`.

## Decisions taken before the final test

- The baseline configuration was evaluated on the validation set and
  selected using the validation PR-AUC checkpoint policy, without
  inspecting the held-out test set at any point during training.
- No hyperparameter search or alternative architectures were compared:
  this project contains a single baseline run, so no selection between
  competing configurations was performed.

## Final test results (frozen validation threshold, one prediction pass)

- Test accuracy: **0.8045 (80.45 %)**, versus the training-majority
  baseline of 0.6250 (62.50 %).
- Confusion counts: **false negatives = 0, false positives = 122**.

## Observations

- Zero false negatives means every PNEUMONIA case in the held-out test
  set was flagged by the model (recall 1.00 at the frozen threshold).
  The cost of this behaviour is the 122 false positives: NORMAL images
  predicted as PNEUMONIA, which lowers the positive predictive value.
- The validation estimates (AP 0.9992, F1 0.9865) are substantially
  stronger than the final test results. The validation pool was
  constructed from the original `train` + `val` folders, while the test
  set remained an untouched source split. This distributional difference
  is consistent with the observed gap, but the cause cannot be established
  from a single run. The test figures are the final estimate of
  generalization and are the numbers reported in the README.
- The result characterizes this single configuration; it is not a claim
  about the best achievable performance on this dataset, since no
  competing configurations were run.

## Reproducibility

- Seed: 42. Exact package versions: `results/environment.json`.
  Frozen run configuration: `results/run_config.json`.
- The trained checkpoint (`best_cnn.keras`) and the frozen
  `inference_config.json` are stored in the Drive output folder of the
  selected run and are not committed to this repository.
