# CNN stage complete

Full ResNet-18 from-scratch experiment completed: 60 epochs x seeds 42,123,2026 on the exact shared split 45,000/5,000/10,000.

Test accuracy mean/std: 93.8433% +/- 0.2104 percentage points. Macro-F1: 0.9384718592 +/- 0.0021709452. SVM accuracy: 57.45%; CNN mean improvement: 36.3933 percentage points.

Per seed (best validation epoch, test accuracy):
- 42: epoch 60, 93.80%.
- 123: epoch 59, 93.61%.
- 2026: epoch 58, 94.12%.

Mean training wall per seed: 413.10 seconds (6m53s), including validation and checkpoint I/O. All three training runs: 1239.29 seconds (20m39s). Core epochs after startup take about 6.2 seconds. Maximum measured allocated VRAM: 2533.20 MiB (2.47 GiB); later warm runs peaked at 858.51 MiB. Report the maximum for capacity planning; allocator peak includes resident training images and startup/autotuning. Mean batch-512 end-to-end inference: 0.09466 ms/image on GPU; SVM: 0.54066 ms/image on CPU. Different models/devices; this is not a same-algorithm hardware speedup.

11,173,962 trainable parameters; best checkpoint 44,776,523 bytes per seed. All weights, optimizer/scheduler/RNG last checkpoints, histories, metrics, predictions, resource measurements, configuration, source snapshots and plots are preserved in results/20261111_005816_cnn_resnet18/.

Verification:
- All 10 behavioral/regression checks passed.
- All 30,000 prediction rows audited, saved metrics and mean/std independently recomputed, checkpoint/data/split hashes unchanged.
- Fresh reload: 29,999 exact matches and one explicitly recorded top-two numerical near-tie; accuracy unchanged. See NUMERICAL_VERIFICATION.md. No assertion of bitwise GPU equivalence.
- Image-file prediction command returned the expected cat label for the saved example.
- Validation-max checkpoint selection verified for every seed; all selections froze before test evaluation.
- Original SVM independently reverified unchanged.
- Four final plots visually inspected and readable.
- Review findings about interrupted-run checkpoint consistency and peak-memory carryover were fixed RED -> GREEN in root code; active run was uninterrupted and used its preserved original source. See code/TRAINING_CODE_NOTES.md and code/recovery_fixes.patch.

Sources: four recent research PDFs plus the foundational ResNet paper, with bibliography and SHA-256 manifest.

Next planned stage: pretrained vision transformer; final combined report and six-slide presentation follow all three models.
