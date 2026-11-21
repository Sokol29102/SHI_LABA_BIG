# Transformer completion

## Completed 2026-11-11

All three models, source archive, Ukrainian 12-page report and six-slide defense are complete. Final outputs are in `deliverables/`.

Test accuracy: SVM 57.45%; ResNet-18 93.84% ± 0.21 percentage points; DeiT-Tiny 97.14% ± 0.07 percentage points (neural mean ± std over three seeds). DeiT uses ImageNet-1K pretraining and 224x224 input; local timing excludes upstream pretraining.

Transformer run: `results/20261111_023428_transformer_deit`. All 30,000 transformer predictions reproduced exactly in fresh-process verification. CNN retains the documented one-label near-tie reload limitation; its accuracy is unchanged. All 13 unit tests passed. Prediction CLI on the saved example returned `cat`. Final matched inference benchmark uses batch 512 and strict FP32 for both neural models, TF32 off; SVM runs on CPU. This differs from the earlier TF32-allowed CNN timing.

Presentation structure/native chart checks passed. All six slide renders and report figures were visually reviewed. PPTX remains editable; slide PDF is a visual copy generated from the checked renders. Speaker notes contain 385 spoken words for approximately 3–5 minutes.
