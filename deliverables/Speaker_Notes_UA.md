# Нотатки для захисту на 3-5 хвилин

## Слайд 1

Я порівнюю три способи розпізнавання тих самих десяти класів: класичний SVM, згорткову мережу ResNet-18 і vision transformer DeiT-Tiny. У CIFAR-10 є шістдесят тисяч маленьких зображень. Для всіх моделей збережено однакові сорок п’ять тисяч training-прикладів, п’ять тисяч validation і десять тисяч test. Мета роботи - не просто знайти найбільшу accuracy, а виміряти, якими часом, пам’яттю та складністю вона досягається.

Джерела: CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html ; власні data/audit.json, data/split_indices.npz.

## Слайд 2

Для огляду я використав чотири реальні роботи 2024-2025 років і зберіг їхні PDF та бібліографію. SHViT і TinyNeXt показують, що пам’ять і набір операцій важливі поряд із кількістю параметрів. MobileNetV4 підкреслює залежність швидкості від конкретного пристрою та попереднього навчання. Робота Wu безпосередньо досліджує невеликий ViT на CIFAR-10, але це препринт з іншим протоколом. Тому результати цих авторів є контекстом, а не прямою таблицею рекордів для нашого запуску.

Джерела: [1] https://arxiv.org/abs/2501.06220 ; [2] https://arxiv.org/abs/2401.16456 ; [3] https://arxiv.org/abs/2404.10518 ; [4] https://www.openaccess.thecvf.com/content/ICCV2025/papers/Zeng_An_Efficient_Hybrid_Vision_Transformer_for_TinyML_Applications_ICCV_2025_paper.pdf

## Слайд 3

SVM працює на фіксованих HOG і RGB-гістограмах. Scaler та параметр C підбиралися тільки всередині training cross-validation. ResNet-18 навчався з нуля шістдесят епох. DeiT-Tiny отримав ImageNet backbone і новий десятикласовий head, після чого всі параметри донавчалися десять епох. Для нейромереж використано три seeds замість повної CV. Кращі checkpoint обрано за validation macro-F1 і зафіксовано до test. Pretraining і різна роздільність є важливими відмінностями процедур.

Джерела: Власні config.json та all_selections_frozen_before_test.json. ResNet: https://arxiv.org/abs/1512.03385 ; DeiT: https://arxiv.org/abs/2012.12877 ; pinned weights: https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k

## Слайд 4

На тому самому test SVM дав 57,45 відсотка accuracy. ResNet-18 - 93,84 плюс-мінус 0,21 процентного пункта. DeiT-Tiny - 97,14 плюс-мінус 0,07. Отже, середній виграш transformer над CNN становить 3,30 процентного пункта. Macro-F1 підтверджує той самий порядок. Це середні результати трьох запусків, без вибору найкращого seed за test. SVM має лише один фінальний test-результат.

Джерела: Власні comparison_all_models.csv, metrics.json та predictions CSV трьох експериментів.

## Слайд 5

Ціна якості залежить від того, що саме вимірюємо. Локальна адаптація DeiT зайняла близько 5,1 хвилини на seed, CNN - 6,9. Але ImageNet pretraining не входить у ці числа. Для inference обидві нейромережі порівняно у strict FP32 на однаковому GPU, batch 512. DeiT потребує приблизно 1,8 раза більше часу на зображення, хоча параметрів у нього менше. SVM працює на CPU, тому це не ізольований hardware speedup одного алгоритму.

Джерела: Власні matched_inference_benchmark.json, training_resources.json, comparison_all_models.csv. ms/image - batch throughput, не single-request latency.

## Слайд 6

Для максимальної якості в цьому експерименті доцільний fine-tuned DeiT-Tiny. Для меншого measured GPU latency - ResNet-18. SVM залишається компактним CPU baseline, але його якість значно нижча. Водночас це не чистий тест attention: transformer має ImageNet pretraining і вхід 224 на 224. Дані 32 на 32 залишають неоднозначні зображення, що видно в помилках. Усі checkpoint та прогнози збережено. CNN мав одну числову near-tie при reload без зміни accuracy; для DeiT у strict FP32 підтверджено всі тридцять тисяч прогнозів.

Джерела: Власні per_class_metrics.csv, confusion matrices та verification.json. Обмеження числової відтворюваності: https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html
