# GPU reload numerical audit

The experiment used BF16 autocast training, FP32 evaluation with TF32 enabled, cuDNN benchmarking, and explicitly did not promise bitwise GPU determinism (see config.json).

The first exact-label reload check exposed one differing prediction. Diagnostics isolated seed 123, original test index 3828: saved prediction truck (9), fresh prediction airplane (0), true label automobile (1). Both predictions are incorrect, so accuracy is unchanged. Fresh top-two logit gaps were approximately 0.000263 in diagnosis and 0.000103 in the final verification. Checkpoint and dataset SHA-256 hashes are unchanged.

The final verifier independently recomputes recorded accuracy/F1 and aggregate mean/std from the saved prediction rows. Fresh GPU predictions must either match exactly or swap only the two highest-scoring classes with logit gap <=0.001. Every such exception is recorded; larger gaps or unrelated-label changes fail. This is an explicit near-tie audit criterion, not a universal floating-point error bound or a bitwise-reproducibility guarantee. Regression checks reject high-confidence and unrelated-label changes.

Final audit: 30,000 prediction rows checked; 29,999 exact reload matches, one listed near-tie. Recorded and fresh accuracy are identical for every seed. The one changed label alters seed-123 macro-F1 by about 0.000001; recorded experiment metrics and all checkpoints are preserved. Strict-FP32 diagnostic evaluation also differed by 1-2 labels per 10,000 relative to the original fast evaluation; those diagnostic values do not replace the recorded experiment.

Reference: PyTorch 2.11 numerical accuracy documentation explains floating-point non-equivalence and TF32 precision: https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html .
