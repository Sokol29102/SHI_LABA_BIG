# Третя модель: fine-tuning DeiT-Tiny

Test accuracy: 97.14% ± 0.07 п.п.; macro-F1: 0.9714 ± 0.0007.

Локальне навчання: 10 епох x 3 seeds; середній час: 303.2 с/seed. Backbone попередньо навчено на ImageNet-1K; його початкова вартість не вимірювалася і не включена. Це порівняння практичних процедур, а не ізольований доказ переваги attention.

Train/validation/test незмінні. Усі checkpoint обрано за validation macro-F1 та зафіксовано до test. 32x32 збільшується до 224x224 bicubic без додавання деталей; використано published ImageNet normalization. Transformer evaluation - strict FP32, TF32 off. Спільний inference benchmark для обох нейромереж також strict FP32; CPU SVM має інший hardware. Якість CNN збережено з його початкового FP32/TF32-allowed запуску з документованим near-tie caveat. Seed42 наперед обрано для ілюстрацій і latency, без вибору seed за test.

Детальні quality/time/RAM/VRAM/parameters/storage/latency дані: comparison_all_models.csv, metrics.json, matched_inference_benchmark.json. SVM RAM peak охоплює CV workers; neural RAM/VRAM включають resident dataset і можуть залежати від startup/autotuning. Std SVM по folds не прирівнюється до neural seed std; test seed std SVM не оцінено і залишено порожнім у CSV.
