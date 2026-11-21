# Training code provenance

The active uninterrupted GPU experiment started with code/train_cnn.py. That exact source snapshot is preserved and must not be overwritten.

During training, review found two issues confined to interrupted-run recovery: exported best.pt could disagree with an older committed last.pt, and pre-interruption memory peaks were not retained. Both were regression-tested RED -> GREEN and fixed in the project-root train_cnn.py for future/resumed processes. The active process was neither restarted nor hot-patched. Its normal-path selection, training and resource measurements remain those of the preserved source.

The root version additionally keeps the authoritative best weights and cumulative resource peaks in last.pt, and rejects inconsistent legacy recovery. Incomplete peak history from a legacy resume is explicitly marked as a lower bound. Future checkpoint I/O cost may therefore differ slightly. Both versions share the same architecture, data, augmentation, loss, optimizer, scheduler and validation/test protocol.

The future-recovery changes are recorded separately as recovery_fixes.patch. Prediction/verification scripts and helper-module snapshots are added when the experiment finishes. Test results and model hashes are independently checked after all seeds finish.
