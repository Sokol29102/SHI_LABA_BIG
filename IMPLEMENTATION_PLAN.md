# CIFAR-10 classical baseline implementation plan

> Execution: implement directly in this session using executing-plans and verify before completion.

Goal: download the complete official CIFAR-10 dataset, train the first classical model, and preserve usable artifacts in E:\labs\SHI_LABA_BIG.
Spec: LAB_DESIGN_UA.md, narrowed by the user's current request to dataset and model 1.

Architecture: a preparation script verifies the official archive and builds shared immutable split indices; one baseline module handles fixed HOG/color features and leakage-safe SVM fitting; a small prediction script uses the saved bundle. No notebook, server, neural training, report or slides in this stage.
Stack: Python 3.10, NumPy, scikit-image, scikit-learn, joblib, matplotlib, psutil. Separate .venv; freeze resolved package versions.

Global constraints: seed 42; official test stays held out; stratified validation approximately 10% of cleaned official train; identical image hashes cannot cross any split or CV fold; three-fold CV inside train only; choose C and optional color features by CV macro-F1; final refit uses train only, matching later models. Preserve validation. No invented results.

Review focus: hash duplicates with conflicting labels; train-only scaler fitting; feature extraction consistency after reload; interrupted download/run preservation; predictions and saved metrics agreement.

## Tasks
- [x] 1. Add a small unittest file with real fixtures for duplicate removal/conflicting labels, reproducible disjoint splits, feature shape and finite values, and a fitted model round-trip. Run before implementation (expected missing module).
- [x] 2. Implement prepare_data.py: verified 163 MB official archive, dataset.npz, split indices, duplicate audit, class counts, original indices, provenance and checksums. Save data before training. Download with .part then rename after MD5 matches.
- [x] 3. Implement baseline.py and train_baseline.py: fixed HOG plus optional RGB histogram, three-fold stratified CV over C=[0.0001,0.001,0.01] and HOG/HOG+color; pipeline fits scaler within each fold; cap fit parallelism at 3 and native threads at 1. Save CV rows/folds, selected settings, final fitted bundle, validation/test metrics, predictions, timings, sampled peak process-tree RAM, coefficient count, model size, confusion matrices and illustrative errors. Save a run manifest/status and refuse to overwrite a completed run. Full run uses all cleaned training data, no subsampling.
- [x] 4. Implement predict.py and Ukrainian README with exact commands, output meanings, resource measurement limits, dataset citation and license caveat. Save dependency lock via pip freeze.
- [x] 5. Run the tests; prepare full data; run complete CV/refit/evaluation. Reload model in a new process, independently recompute saved test metrics and confirm predictions. Inspect PNGs and artifacts. Update progress ledger with actual counts/results.

No git repository exists in the requested lab folder; do not initialize or alter unrelated repositories. User explicitly authorized implementation of this stage; no additional design approval is needed.

