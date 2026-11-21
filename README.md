# Лабораторна: CIFAR-10, classical ML baseline

Перша модель: фіксовані HOG ознаки (1764) та опційні RGB гістограми (96), StandardScaler і LinearSVC. Notebook не потрібний. Моделі CNN та transformer належать до наступного етапу.

## Запуск у PowerShell

```powershell
Set-Location E:\labs\SHI_LABA_BIG
& .\.venv\Scripts\python.exe -m unittest -v test_baseline
& .\.venv\Scripts\python.exe .\prepare_data.py
& .\.venv\Scripts\python.exe -u .\train_baseline.py
```

Окреме `.venv` не змінює середовище іншої лабораторної. Для відтворення середовища:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

## Дані й чесне порівняння

Завантажується повний офіційний CIFAR-10 (60 000 зображень, 10 класів); архів перевіряється за офіційним MD5. Посилання: https://www.cs.toronto.edu/~kriz/cifar.html . Цитування: Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009.

На офіційній сторінці не встановлено явної ліцензії dataset; зазначаємо джерело й не додаємо дані до архіву для поширення.

`data/dataset.npz` містить оригінальні масиви. `data/split_indices.npz` містить незмінні індекси train/validation/test для всіх трьох майбутніх моделей. Train/validation формуються з очищеної офіційної train-частини 90/10, seed 42. `data/audit.json` зберігає точні розміри, баланс класів і всі вилучені індекси з причинами.

У кожній офіційній частині залишаємо перший точний дублікат; групи однакових зображень із суперечливими labels вилучаємо. З train також вилучаємо будь-яке зображення, SHA-256 якого зустрічається в офіційній test-частині. Це QC за наперед визначеним правилом, а не підбір за test-результатами. Так усувається exact leakage; схожі кадри, crop і flip не усуваються автоматично. Через очищення оцінка може не бути прямо порівнянною з published unfiltered benchmark.

3-fold CV тільки всередині train вибирає C із [0.0001, 0.001, 0.01] і HOG або HOG+RGB за macro-F1. Scaler fit входить у кожен fold. Після вибору виконується refit лише на спільному train. Validation зберігається незалежним для наступних моделей. Перед test оцінюванням параметри й модель уже збережені.

## Що збережено

Кожен запуск має окрему папку `results/<дата>_classical_svm/`:

- `model.joblib`: sklearn pipeline, назви класів, схема ознак і вибрані параметри.
- `metrics.json`, `summary.csv`, `per_class_metrics.csv`: фактичні метрики й ресурси.
- `cv_results.csv`, `cv_fold_indices.npz`: кожна конфігурація та оригінальні індекси folds.
- `test_predictions.csv`, `validation_predictions.csv`: прогноз кожного оціненого зображення.
- `config.json`, `selected_config_before_test.json`, `status.json`, `code/`, `run.log`: конфігурації, provenance, копія коду та стан запуску.
- `confusion_matrix*.png`, `cross_validation.png`, `error_examples.png`: графіки.
- `RESULTS_UA.md`: короткий опис фактичного результату.

`results/latest_classical.txt` вказує тільки на останній повністю успішний запуск. Попередні результати не перезаписуються. Кеш ознак зберігається в `data/features/`; це stateless preprocessing без fit на labels.

Час фінального fit подається окремо від повного CV. Cold feature extraction зберігається навіть при повторному використанні кешу. Peak RAM вимірюється як сума RSS parent/workers кожні 100 ms: включає дані й може двічі рахувати shared pages. Classical модель працює на CPU. Inference включає HOG, RGB, scaler та SVM; читання/resize зовнішніх файлів до нього не входить. Batch throughput не тотожний latency одного зображення.

## Нові зображення

```powershell
$run = Get-Content .\results\latest_classical.txt
& .\.venv\Scripts\python.exe .\predict.py --model "$run\model.joblib" .\my_image.png
```

Зовнішнє зображення перетворюється на RGB 32x32. Модель завжди обирає один із 10 класів, навіть для невідомих об’єктів. Завантажуйте лише власні довірені joblib-файли.

Дослідження, PDF, бібліографія та джерела: `sources/RELATED_WORK_UA.md`.

## Перевірка збереженого результату

```powershell
& .\.venv\Scripts\python.exe .\verify_results.py
```

Перевіряє SHA-256 моделі, dataset та split, ізоляцію CV, усі 10 000 прогнозів після reload, метрики та PNG. Фактичний завершений запуск: `results/20261111_002944_classical_svm/`; test accuracy 57.45%, macro-F1 0.5672. Деталі у `PROGRESS.md` та `RESULTS_UA.md` у папці запуску.

## Друга модель: ResNet-18 на GPU

Без pretrained weights, CIFAR stem, той самий train/validation/test. 60 епох для seeds 42, 123, 2026; checkpoint за validation macro-F1, усі три вибори зафіксовано до відкриття test. Нормалізація обчислена лише по train. GPU BF16 використовується тільки для training; evaluation - FP32.

```powershell
Set-Location E:\labs\SHI_LABA_BIG
& .\.venv\Scripts\python.exe -u .\train_cnn.py --epochs 60
& .\.venv\Scripts\python.exe .\verify_cnn.py
```

У `results/<дата>_cnn_resnet18/` зберігаються `config.json`, `run.log`, `seed_42/`, `seed_123/`, `seed_2026/`, `best.pt`, `last.pt`, історія кожної епохи, predictions, метрики та графіки. `results/latest_cnn.txt` оновлюється лише після повного успішного запуску. `best.pt` - модель з найкращим validation macro-F1; `last.pt` також містить optimizer/scheduler/RNG для відновлення.

```powershell
& .\.venv\Scripts\python.exe .\train_cnn.py --resume .\results\<папка_перерваного_CNN_запуску>
$cnnRun = Get-Content .\results\latest_cnn.txt
& .\.venv\Scripts\python.exe .\predict_cnn.py --model "$cnnRun\seed_42\best.pt" .\my_image.png
```

Для CPU inference додайте `--device cpu`. Для відтворення CUDA-середовища використовуйте `requirements-cnn-lock.txt` і офіційний PyTorch CUDA 12.8 index; локальні wheels встановлено в окреме середовище цієї лабораторної. Попередній SVM lock-файл збережено.

CNN та SVM використовують різні пристрої й training procedures; порівняння їхнього часу не є вимірюванням GPU-vs-CPU speedup того самого алгоритму. У CNN std по seeds, у SVM CV std по folds; ці оцінки нестабільності не тотожні.

CNN завершено: test accuracy 93.84% +/- 0.21 п.п. для трьох seeds; усі артефакти й GPU вимірювання у `results/20261111_005816_cnn_resnet18/`. При перевірці reload одна з 30 000 міток змінилася через top-2 near-tie; accuracy не змінилася. Обмеження і точний критерій аудиту зафіксовані у `NUMERICAL_VERIFICATION.md` та `verification.json` цієї папки. Це не гарантія bitwise GPU equivalence.

## Третя модель та готові матеріали

DeiT-Tiny: ImageNet-1K pretrained backbone, immutable revision `80e968688553f219e4a86f940ed945a23709c16f`, усі параметри fine-tuned. 10 епох × seeds 42, 123, 2026. BF16 training, strict FP32 evaluation, TF32 off. Вхід 224x224 отримано bicubic resize з CIFAR 32x32. Повний lock: `requirements-transformer-lock.txt`. Provenance: `data/pretrained/deit_tiny/provenance.json`.

```powershell
Set-Location E:\labs\SHI_LABA_BIG
& .\.venv\Scripts\python.exe .\train_transformer.py --help
& .\.venv\Scripts\python.exe .\verify_transformer.py
$vitRun = Get-Content .\results\latest_transformer.txt
& .\.venv\Scripts\python.exe .\predict_transformer.py --model "$vitRun\seed_42\best.pt" .\my_image.png
```

Для відновлення перерваного запуску: `train_transformer.py --resume <папка_запуску>`. `best.pt` зберігає вибраний validation checkpoint, `last.pt` також optimizer/scheduler/RNG і committed best state. Для inference мережа та upstream weights не потрібні. Для CPU додайте `--device cpu`.

Готові матеріали у `deliverables/`: `Lab_Report_UA.pdf` (12 сторінок), редаговане джерело `Lab_Report_UA.md`, `Lab_Defense_6slides_UA.pptx` (6 редагованих слайдів), PDF-копія слайдів, `Speaker_Notes_UA.md` (3–5 хв), `Comparison_3models.csv`. Джерела: `sources/` — чотири дослідження 2024–2025 та дві фундаментальні праці, оригінальні PDF, бібліографія й SHA-256.

Слайдовий PDF є візуальною копією перевірених renders. Нативний графік і таблиці редагуються в PPTX. Код побудови: `build/presentation/build_slides.mjs`, `prepare_deliverables.py`, `render_report.py`, `render_slides_pdf.py`.
