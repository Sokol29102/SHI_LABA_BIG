# Дизайн лабораторної: класифікація зображень CIFAR-10

Статус: пропозиція для погодження перед реалізацією.
Дата: 2026-11-11.
Робоча папка: E:\labs\SHI_LABA_BIG.

## Мета та результат
Порівняти класичне ML, CNN і vision transformer на одній задачі розпізнавання 10 класів. Пояснити приріст якості відносно часу, пам’яті та складності. Усі числові результати отримати фактичним запуском на цьому комп’ютері. Не вигадувати показники й не подавати коротку перевірку за повний експеримент.

Користувач обрав зображення, локальне виконання та презентацію приблизно на 6 слайдів для кількох хвилин. Notebook не є обов’язковим; основний інтерфейс - Python scripts та README. Обов’язково використати реальні дослідження й зберегти першоджерела. Решта нижче - запропоновані технічні рішення.

## Розглянуті підходи
- CIFAR-10: 60 000 зображень, 10 збалансованих класів, компактне завантаження, достатньо прикладів для CNN. Рекомендований варіант.
- Imagenette: зображення більшої роздільності та природний вхід для pretrained ViT; менше прикладів. Також придатний, незалежно від використання в іншому предметі.
- Класифікація новин: технічно придатна, але користувач обрав зображення.

## Дані та протокол
Джерело: https://www.cs.toronto.edu/~kriz/cifar.html
Офіційні 50 000 train та 10 000 test. З train сформувати стратифіковані train/validation 45 000/5 000, seed 42. Перевірити пошкоджені об’єкти, класи, баланс, exact duplicates і перетин split; вилучення або групування дублікатів задокументувати з фактичними новими розмірами до навчання. Близькі дублікати дослідити й описати як обмеження, не обіцяючи їх повного усунення. Зберегти індекси та контрольні суми.

Усі моделі використовують ті самі об’єкти train/validation/test. Нормалізацію, масштабування та вибір параметрів виконувати без test. Перетворення входу залежать від моделі й описуються окремо. Test відкривається для фінального оцінювання лише після фіксації конфігурацій.

Трикратна cross-validation для classical ML тільки всередині train; scaler входить у pipeline кожного fold. Для нейромереж використовувати фіксований validation та seeds 42, 123, 2026 замість повної CV, з поясненням обчислювального обмеження. Для classical ML показати mean/std validation CV, після вибору C виконати один refit на всьому train й отримати один фінальний test-результат без штучного test std. Для нейромереж показати mean/std test за трьома seeds після фіксації всіх конфігурацій. У підсумковій таблиці явно розділити CV validation та test; не прирівнювати std CV до std різних seeds. Не обирати найкращий seed за test.

Перевірити умови використання та джерело CIFAR-10; у звіті чесно зазначити, якщо явна ліцензія не встановлена. Не включати сам dataset до фінального архіву.

## Три моделі
1. HOG + StandardScaler + LinearSVC. Додати прості колірні характеристики, якщо validation покаже користь; невелика наперед визначена сітка C. Обчислення HOG та inference враховуються в ресурсній ціні.
2. ResNet-18 зі CIFAR-сумісним stem, без pretrained weights. Аугментації тільки train; вибір checkpoint за validation macro-F1.
3. DeiT-Tiny, pretrained ImageNet-1k, fine-tuning класифікатора та backbone. Вхід 224x224; заміна голови на 10 класів. Джерело: https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k .

Попереднє навчання DeiT та різна роздільність є частиною практичного порівняння процедур, а не контрольованим доказом переваги attention. Вартість ImageNet pretraining не вимірюється цим експериментом і зазначається окремо. Resize 32x32 до 224x224 не додає деталей і збільшує обчислювальну ціну.

## Виконання та вимірювання
Підтверджено RTX 5060 Ti, 16 GB VRAM, працездатний CUDA PyTorch 2.11.0+cu128. Створити окреме середовище проєкту, не змінюючи середовище іншої лабораторної. Остаточні batch size, кількість епох і оцінку тривалості визначити після короткого профілювання тільки train/validation; зафіксувати до повного запуску. Встановити розумний ліміт CPU workers для Windows.

Accuracy, macro Precision/Recall/F1, per-class metrics, confusion matrices; час preprocessing, навчання та inference; peak process RAM і CUDA allocated/reserved memory; розмір артефакту, загальна/навчувана кількість параметрів. Синхронізувати CUDA для таймінгу, прогріти inference, зазначити batch/device. Повний inference включає перетворення входу; окремо можна подати model-only timing. Не оцінювати спожиту енергію або гроші без відповідних вимірювань.

## Огляд робіт
Прочитати 3-4 актуальні первинні статті про ефективні CNN/vision transformers та transfer learning. Для кожної: дата, посилання, dataset, split, модель, метрики, результати, ресурси, відмінності протоколів. Відібрано Wu (2025), SHViT (CVPR 2024), MobileNetV4 (ECCV 2024), TinyNeXt (ICCV 2025). PDF збережено у sources; огляд - sources/RELATED_WORK_UA.md, бібліографія - sources/references.bib, контрольні суми - sources/manifest.json. Історичні ResNet/DeiT - додаткові архітектурні посилання, а не заміна сучасного огляду. Результати авторів не подавати як власні або прямо порівнянні без перевірки умов.

## Фінальні матеріали
- Відтворювані Python scripts для підготовки даних, навчання, оцінювання та побудови графіків; зафіксовані залежності та README українською. Notebook не є обов’язковим результатом.
- Збережені конфігурації, журнали, моделі, індекси split, CSV/JSON з фактичними результатами.
- Приклади даних, learning curves, confusion matrices, графіки якість/час і якість/пам’ять, аналіз типових помилок.
- Технічний звіт українською у PDF та редагованому форматі: постановка, дані, related work, методологія, три моделі, результати, аналіз, висновки, обмеження, джерела.
- PPTX і PDF презентації, рівно 6 слайдів, приблизно 3-5 хвилин, короткі українські нотатки доповідача.

## Шість слайдів
1. Задача, CIFAR-10, приклади класів.
2. 3-4 сучасні роботи: основний висновок і контекст.
3. Спільний протокол та метрики.
4. Три моделі й важливі відмінності.
5. Фактичні результати: компактна таблиця та графік quality/cost.
6. Помилки, висновок, обмеження та практичний вибір моделі.

## Перевірка готовності
Фактично виконати всі три моделі, перевірити ізоляцію split, коректність метрик і повторюваність збережених конфігурацій. Перевірити команди запуску Python scripts і відновлення checkpoint. Візуально відрендерити PDF та всі шість слайдів, перевірити читабельність, українські символи та відсутність переповнення. Якщо щось перервано або недоступне, чітко позначити це, а не заявляти повне завершення.


## Completed 2026-11-11

All three models, source archive, Ukrainian 12-page report and six-slide defense are complete. Final outputs are in `deliverables/`.

Test accuracy: SVM 57.45%; ResNet-18 93.84% ± 0.21 percentage points; DeiT-Tiny 97.14% ± 0.07 percentage points (neural mean ± std over three seeds). DeiT uses ImageNet-1K pretraining and 224x224 input; local timing excludes upstream pretraining.

Transformer run: `results/20261111_023428_transformer_deit`. All 30,000 transformer predictions reproduced exactly in fresh-process verification. CNN retains the documented one-label near-tie reload limitation; its accuracy is unchanged. All 13 unit tests passed. Prediction CLI on the saved example returned `cat`. Final matched inference benchmark uses batch 512 and strict FP32 for both neural models, TF32 off; SVM runs on CPU. This differs from the earlier TF32-allowed CNN timing.

Presentation structure/native chart checks passed. All six slide renders and report figures were visually reviewed. PPTX remains editable; slide PDF is a visual copy generated from the checked renders. Speaker notes contain 385 spoken words for approximately 3–5 minutes.
