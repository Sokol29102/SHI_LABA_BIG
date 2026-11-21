# Друга модель: ResNet-18 з нуля

Спільний split: {'train': 45000, 'validation': 5000, 'test': 10000}. Навчання: 60 епох, seeds 42, 123, 2026; без pretrained weights.

- Test accuracy: 93.84% ± 0.21 п.п.
- Test macro-F1: 0.9385 ± 0.0022.
- SVM test macro-F1: 0.5672; різниця: 0.3713.
- Середній training wall / seed: 413.1 с.
- Середній peak CUDA allocated: 1416.7 MiB.

Std обчислено по трьох seeds, а не folds. Кожен checkpoint обрано лише за validation macro-F1; усі три вибори зафіксовано до test. Seed 42 наперед обрано для ілюстрацій, а не за результатами test. Нормалізація використовує лише train; augmentation вимкнено на evaluation. CUDA kernels не гарантують bitwise reproducibility.

Порівняння часу не є GPU-vs-CPU speedup одного алгоритму: різні моделі, пристрої та процедури. CNN training wall включає validation і checkpoint I/O; SVM final refit відокремлено від CV search. End-to-end inference обох моделей використовує batch 512 і виключає читання файлів; CNN також включає host-to-GPU transfer. Peak VRAM містить resident training image buffer; це allocated memory процесу, а не загальна VRAM desktop.

## Перевірка відтворюваності

SHA-256 checkpoint/dataset/split, вибір епох за validation, усі 30 000 записів прогнозів і їхні метрики перевірено незалежно. При GPU reload отримано 29 999 точних збігів і одну near-tie відмінність у seed 123, test index 3828: truck/airplane, обидва помилкові; accuracy не змінилася. TF32 і cuDNN benchmarking увімкнені, bitwise determinism не обіцяється. Verifier дозволяє тільки зафіксовані перестановки top-2 при logit gap <=0.001; решта відмінностей є помилкою. Деталі, межі перевірки й офіційне посилання: NUMERICAL_VERIFICATION.md та verification.json. Оригінальні метрики й ваги не змінено.
