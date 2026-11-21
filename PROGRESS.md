# Progress: first classical model complete

User-authorized stage complete: full dataset downloaded, first model trained, selected model and results saved.

Dataset: official full CIFAR-10, 60,000 RGB images. Archive MD5 matches c58f30108f718f92721af3b95e74349a. No exact duplicate or conflicting-label hash groups found. Shared split: 45,000 train / 5,000 validation / 10,000 test, seed 42. Dataset, indices, class audit and SHA-256 provenance are in data/.

Experiment: 18 full CV fits (3 folds x 3 C values x 2 feature variants) plus one final train-only refit. Selected HOG + RGB histograms + StandardScaler + LinearSVC, C=0.001. Scaler fitting stayed within folds; validation was preserved; test was not used for selection.

Results:
- CV macro-F1: 0.5622776324 +/- 0.0031010876 (fold std).
- Test accuracy: 0.5745 (57.45%).
- Test macro-F1: 0.5671615159.
- Validation accuracy: 0.5796.
- Final refit: 133.08 seconds; CV plus refit: 996.29 seconds.
- Peak aggregate process-tree RSS: 7695.61 MiB, sampled every 100ms (shared-page overcount possible).
- Saved model: 186,965 bytes. CPU model; no GPU used.
- End-to-end batch inference: 0.54066 ms/image (HOG/RGB/scaling/SVM, excludes file decoding).

Saved run: results/20261111_002944_classical_svm/ . Latest complete run pointer: results/latest_classical.txt.

Verification:
- All 3 behavioral tests passed after completion.
- Fresh process reloaded model.joblib and reproduced all 10,000 test predictions.
- Independent accuracy, macro-F1 and confusion-matrix recomputation matched saved results.
- Raw-image feature extraction and prediction checked on 64 images.
- CLI prediction on a normal PNG succeeded and matched the expected saved prediction.
- Shared split hash, model hash, CV fold isolation and original test labels verified.
- PNG files decoded successfully; normalized confusion matrix, CV plot and random errors visually inspected and readable.
- Independent code review found no important actionable bugs.

Ruling: work directly in the user-designated folder; no repository setup or changes to other labs.
Ruling: current authorization covers data and model 1. CNN/transformer training, final report and six-slide presentation are later stages.
Environment note: a sandbox-only test rerun could not write a temporary file; the same tests passed with the lab's authorized filesystem access, without changing code.

# Stage 2 update: CNN complete

ResNet-18, 60 epochs x three seeds: test accuracy 93.84% +/- 0.21 p.p.; macro-F1 0.9385 +/- 0.0022. Full artifacts: results/20261111_005816_cnn_resnet18/ . See CNN_PROGRESS.md for GPU resource measurements and the explicit numerical reload caveat. Ten checks passed, recorded metrics verified, original SVM preserved. Transformer, full report and six-slide presentation remain subsequent stages.

## Completed 2026-11-11

All three models, source archive, Ukrainian 12-page report and six-slide defense are complete. Final outputs are in `deliverables/`.

Test accuracy: SVM 57.45%; ResNet-18 93.84% ± 0.21 percentage points; DeiT-Tiny 97.14% ± 0.07 percentage points (neural mean ± std over three seeds). DeiT uses ImageNet-1K pretraining and 224x224 input; local timing excludes upstream pretraining.

Transformer run: `results/20261111_023428_transformer_deit`. All 30,000 transformer predictions reproduced exactly in fresh-process verification. CNN retains the documented one-label near-tie reload limitation; its accuracy is unchanged. All 13 unit tests passed. Prediction CLI on the saved example returned `cat`. Final matched inference benchmark uses batch 512 and strict FP32 for both neural models, TF32 off; SVM runs on CPU. This differs from the earlier TF32-allowed CNN timing.

Presentation structure/native chart checks passed. All six slide renders and report figures were visually reviewed. PPTX remains editable; slide PDF is a visual copy generated from the checked renders. Speaker notes contain 385 spoken words for approximately 3–5 minutes.
