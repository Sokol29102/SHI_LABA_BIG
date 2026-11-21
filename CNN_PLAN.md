# CIFAR-10 ResNet-18 implementation plan

Goal: execute model 2 from LAB_DESIGN_UA.md, preserving the completed SVM and identical shared splits. User authorized this stage with “let's do next”.
Execution: native inline, following executing-plans, TDD and verification-before-completion. No additional design approval: architecture and three-seed protocol were already specified. No notebook or transformer work in this stage.

Architecture: torchvision ResNet-18, weights=None, CIFAR stem (3x3, stride 1, no maxpool), 10 classes. Train-only computed RGB mean/std. Random reflection-padded crop and horizontal flip only during training, implemented per image on GPU. Resident uint8 image tensors avoid a Windows DataLoader bottleneck; preprocessing costs and resident VRAM are measured. BF16 autocast training; FP32 validation/test. SGD, momentum 0.9, Nesterov, weight decay 5e-4, cosine decay. Initial batch 256 and LR 0.1; 60 epochs per seed proposed, freeze after train-only timing profile. Seeds 42, 123, 2026. Select each best checkpoint by validation macro-F1; freeze all selections before any test evaluation. Report all seeds and mean/std, never select seed on test.

Files: cnn.py (model, normalization, preprocessing and checkpoint loading), train_cnn.py (profile, three-seed experiment, saved histories/checkpoints/metrics/plots/comparison), test_cnn.py (behavioral checks), verify_cnn.py (fresh-process verification), predict_cnn.py (ordinary image inference). Reuse train_baseline.py metrics, CSV and memory sampler; prepare_data.py hash/provenance helpers.

Review focus: no validation/test influence on statistics or checkpoint selection; eval cannot mutate BatchNorm or enable augmentation; checkpoint load must reproduce predictions; three-seed aggregate must distinguish fold std from seed std; timings require CUDA synchronization and matched inference scopes.

- [x] Install already-downloaded CUDA wheels into this lab's own .venv and freeze dependencies; confirm real GPU forward/backward.
- [x] Write and run failing tests for train-only statistics, normalization/augmentation contract, real CNN parameter update and evaluation/checkpoint round-trip.
- [x] Implement cnn.py and training runner. Profile only training batches with a throwaway model; save actual throughput/VRAM. Freeze final config before full runs.
- [x] Run all three seeds on all 45,000 train images; preserve 5,000 validation. Save history.csv, best.pt and last.pt atomically each epoch, plus optimizer/scheduler/RNG in last.pt to support resume. Do not open test labels until all seed selections are frozen.
- [x] Evaluate frozen checkpoints on the same 10,000 test images. Save every prediction, per-class metrics, confusion matrices, loss/F1 curves, random errors, RAM/VRAM, parameter counts, checkpoint sizes and synchronized model-only/end-to-end latency. Compare with saved SVM, explicitly noting CPU/GPU, batch, CV versus seed std, and training-augmentation differences.
- [x] Fresh-process reload and independent metric recomputation for all seeds; check raw-image prediction command, run the complete test suite, inspect plots, obtain independent read-only code review, update README/progress.

Constraints: keep existing SVM artifacts and requirements snapshot; no test-driven tuning; no pretrained CNN weights; GPU nondeterminism and pretraining differences must be documented. Seed 42 is the predeclared example model for plots/comparison, not the best seed selected on test. Logs/checkpoints are retained after failure; new runs get new folders. User requested E:\labs\SHI_LABA_BIG as sole final project location.

