# DeiT-Tiny fine-tuning plan

Goal: complete model 3 from LAB_DESIGN_UA.md, preserving the SVM/CNN, then compare all three procedures. Native inline execution; user explicitly requested continuation. No new design gate.

Architecture: timm deit_tiny_patch16_224.fb_in1k, ImageNet-1K pretrained backbone, fresh 10-class head, all parameters fine-tuned. Preserve immutable upstream revision and SHA-256 of safetensors/config. Use model's published normalization constants, direct bicubic 32->224 resize (no detail added), train-only per-image padded crop/flip shared with CNN, and no augmentation at evaluation. Report transfer-learning and input-resolution differences explicitly; local costs exclude original ImageNet pretraining.

Protocol: unchanged train/validation/test indices and classes; seeds 42,123,2026. Proposed 10 epochs, batch 128, AdamW backbone LR 5e-5/head LR 5e-4, weight decay .05, gradient clipping 1.0, one-epoch linear warmup then cosine. BF16 training, deterministic strict-FP32 evaluation with TF32 disabled and fixed cuDNN settings. Freeze final budget after train-only GPU timing profile, before test evaluation. Select each seed's earliest maximum validation macro-F1 checkpoint; freeze all three before opening test labels. Never select a seed on test.

Files: vit.py (model/preprocessing/download/eval policy), train_transformer.py (profile, training, atomic best/last checkpoint, resources and histories), transformer_results.py (frozen test evaluation, figures and comparison), verify_transformer.py (fresh-process reload and metrics), predict_transformer.py, test_vit.py. Reuse existing data/hash, CSV, memory sampler, seed/checkpoint, augmentation helpers. Preserve original model scripts.

Review focus: complete pretrained backbone loading with only the head excluded; no network dependency when loading fine-tuned checkpoints; train-only augmentation and immutable split; transactional best checkpoint and cumulative resume resource peaks; exact metric/mean-std and checkpoint-selection verification; explicit different hardware, pretraining, resize, precision, and timing scopes in comparison.

- [x] Install timm, download immutable official pretrained weights/config/model card and foundational DeiT paper; preserve provenance/licenses/checksums and lock resolved dependencies.
- [x] Tests first: real 224x224 preprocessing contract, pretrained-backbone/new-head loading boundary, fresh checkpoint round-trip. Watch initial checks fail, then implement helpers and run all tests.
- [x] Profile a throwaway train-only batch on GPU. Freeze batch/epoch/optimizer/eval configuration and save hardware/software metadata.
- [x] Train three seeds on all 45,000 training images. Save best/last/history every epoch with committed best state, optimizer/scheduler/RNG and cumulative peak measurements. Resume only matching immutable configuration. Preserve selections before test.
- [x] Evaluate all frozen models on 10,000 test images, save predictions/per-class/aggregate metrics, synchronized timings, parameters/model sizes and plots. Compile comparison.csv/figures across the three models without conflating seed std and CV std, CPU and GPU, or pretraining and local training cost.
- [x] Independent read-only code review, fix important findings with regressions, fresh-process exact checkpoint reload and metric recomputation, prediction CLI, full tests and visual plot QA. Update README/progress and preserve source snapshots.

After all models are verified, original requested remaining deliverables are the Ukrainian technical report and approximately six slides for a 3-5 minute defense, using saved real research sources and measured results.
