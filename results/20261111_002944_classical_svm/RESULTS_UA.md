# Перша модель: HOG + Linear SVM

Повний запуск CIFAR-10. Train: 45000; validation: 5000; test: 10000.
Дублікатів між цими split немає за SHA-256; близькі дублікати можуть залишатися. Деталі: data/audit.json.

- Обрано C=0.001; колірні гістограми: True.
- 3-fold CV macro-F1: 0.5623 ± 0.0031 (std по folds).
- Validation accuracy: 0.5796.
- Test accuracy: 0.5745; macro-F1: 0.5672.
- Фінальний refit: 133.08 с; повний CV + refit: 996.29 с.
- Розмір моделі: 0.18 MiB.
- Peak RAM: 7695.6 MiB (100ms sampling, сума RSS процесів; shared pages можуть рахуватися двічі).
- Повний inference без читання файлів: 0.541 ms/зображення, batch 512, CPU.
- GPU для classical ML не використовується.

Гіперпараметри вибрано лише за CV всередині train. StandardScaler fit виконується окремо в кожному fold.
Після фіксації конфігурації модель навчена на всьому спільному train; validation не додано до train.
Test використано для фінальної оцінки, а не вибору параметрів. Це один test-результат, без штучного test std.
Кешовані ознаки не мають навчених параметрів; для нових зображень extract_features повторює ті самі операції.

Сильні/слабкі класи наведено у per_class_metrics.csv, випадкові помилки - error_examples.png.
Це baseline; висновок про перевагу CNN/transformer можливий лише після їхнього навчання.
