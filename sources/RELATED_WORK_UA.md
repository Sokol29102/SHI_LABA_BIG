# Реальні дослідження для лабораторної

Дата перевірки: 11 листопада 2026 року. PDF першоджерел збережено поруч із цим файлом. Це огляд робіт 2024-2025 років, а не твердження про абсолютний SOTA станом на 2026 рік. Усі цифри нижче належать авторам статей; власні вимірювання лабораторної зберігатимуться окремо.

## 1. Gent Wu, 2025
**Powerful Design of Small Vision Transformer on CIFAR10.** arXiv:2501.06220v1. Препринт; підтвердженого peer-reviewed venue на сторінці arXiv немає.

- Сторінка: https://arxiv.org/abs/2501.06220
- PDF: `Wu_2025_Small_ViT_CIFAR10.pdf`; експерименти: розділ 2, таблиці 1-3.
- Дані: CIFAR-10, описано стандартні 50 000 train / 10 000 test. Автор використовує назву Val/Acc; окремий незалежний validation split не встановлено з опису, тому протокол не ототожнюємо з нашим.
- Метод: невеликий ViT, аугментації, low-rank attention та кілька CLS tokens; 100 епох, batch 256, RTX 3090.
- Результат авторів: baseline 93.65%; варіант із двома CLS tokens - 93.95%; при batch 256 таблиця 1 подає 2.50 GB пам’яті.
- Для лабораторної: обґрунтовує важливість аугментацій та вимірювання пам’яті. Це інша модель й інший training recipe, а не очікуваний результат нашого DeiT.

## 2. Seokju Yun, Youngmin Ro, CVPR 2024
**SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design.** arXiv:2401.16456v2.

- Сторінка: https://arxiv.org/abs/2401.16456
- PDF: `Yun_Ro_2024_SHViT.pdf`; розділ 3.1, таблиці 2-3, supplementary B.
- Дані: ImageNet-1K, приблизно 1.28 млн train / 50 тис. validation, 1000 класів.
- Метод: single-head attention та архітектура зі зменшеним memory-access cost; 300 епох з нуля.
- Результат авторів: SHViT-S4 при 256x256 має 79.4% top-1, 16.5 млн параметрів і throughput 14 283 зображень/с на A100, batch 256. Значення 80.2% у таблиці 3 належить distillation recipe.
- Для лабораторної: показувати peak RAM/VRAM і throughput, а не лише кількість параметрів. Час A100 не переносимо на RTX 5060 Ti; ImageNet accuracy не порівнюємо чисельно з CIFAR-10.

## 3. Danfeng Qin та співавтори, ECCV 2024
**MobileNetV4: Universal Models for the Mobile Ecosystem.** arXiv:2404.10518v2.

- Сторінка: https://arxiv.org/abs/2404.10518
- PDF: `Qin_2024_MobileNetV4.pdf`; розділи 7.1 та 8, таблиці 6, 8-10.
- Дані: ImageNet-1K train/validation; окремі distillation експерименти використовують додаткові дані JFT.
- Метод: Universal Inverted Bottleneck, mobile multi-query attention, NAS та distillation; порівняння різних мобільних платформ.
- Результат авторів: MNv4-Conv-S - 73.8% top-1, 3.8 млн параметрів, 2.4 ms на Pixel 6 CPU. MNv4-Hybrid-L у стандартному порівнянні має 83.4%; широко цитовані 87.0% отримано з додатковим pretraining/distillation.
- Для лабораторної: обов’язково вказувати hardware, batch та джерело pretrained weights. Перевагу pretrained transformer не можна приписати лише attention.

## 4. Fanhong Zeng та співавтори, ICCV 2025
**An Efficient Hybrid Vision Transformer for TinyML Applications.** Моделі TinyNeXt.

- Першоджерело: https://www.openaccess.thecvf.com/content/ICCV2025/papers/Zeng_An_Efficient_Hybrid_Vision_Transformer_for_TinyML_Applications_ICCV_2025_paper.pdf
- Код авторів: https://github.com/yuffeenn/TinyNeXt
- PDF: `Zeng_2025_TinyNeXt.pdf`; розділ 4.1, таблиці 2-4.
- Дані: ImageNet-1K; тренування з нуля, 224x224, 300 епох на 8 RTX 3090.
- Метод: hybrid CNN/ViT, Lean Single-Head Self-Attention; оптимізація memory-bound operations.
- Результат авторів: TinyNeXt-T - 71.5% top-1, 1.0 млн параметрів, 7257 зображень/с на RTX 3090, inference batch 512. Для CPU використано AMD EPYC 7542 і batch 16.
- Для лабораторної: latency залежить від конкретних операцій, а не лише FLOPs. Великий inference batch авторів не можна прирівняти до latency одного зображення.

## Як використати у звіті та шести слайдах
У related work порівняти постановку, дані, архітектуру та ресурсні умови. На другому слайді залишити чотири короткі рядки з посиланнями [1]-[4]; повну бібліографію подати у звіті та нотатках. П’ятий слайд міститиме тільки фактично отримані результати нашого експерименту. На шостому зазначити різницю training-from-scratch та transfer learning.

## Збереження джерел
- `references.bib`: бібліографічні записи.
- `manifest.json`: назва, URL, дата отримання, SHA-256, розмір і число сторінок кожного PDF.
- Однойменні `.txt`: допоміжне витягнення для пошуку; у формулах і графіках можливі артефакти extraction. За спірного фрагмента перевіряти PDF.
- Дані статей не підміняють cross-validation і власне тестування моделей.

## Додаткове архітектурне першоджерело: ResNet
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. **Deep Residual Learning for Image Recognition**, CVPR 2016; arXiv:1512.03385 (препринт 2015). https://arxiv.org/abs/1512.03385 . PDF: `He_2016_ResNet.pdf`.

Це історична основа residual architecture, а не заміна чотирьох сучасних робіт вище. У лабораторній використано torchvision ResNet-18 без pretrained weights, з адаптацією stem під CIFAR-10 (3x3 stride 1, без maxpool). Власні результати отримуємо на CIFAR-10, а не повторюємо результати авторів на ImageNet. Офіційна реалізація: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html .
