"""Build report evidence and editable source from verified experimental records."""
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'deliverables'
ASSETS = OUT/'assets'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def fmt(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    paths = {name: Path((ROOT/f'results/latest_{name}.txt').read_text(encoding='utf-8').strip())
             for name in ['classical', 'cnn', 'transformer']}
    for path in paths.values():
        assert read(path/'status.json')['status'] == 'complete'
        assert read(path/'verification.json')['status'] == 'passed'
    models = read(paths['transformer']/'comparison_all_models.json')
    svm, cnn, vit = models
    short = ['HOG + SVM', 'ResNet-18', 'DeiT-Tiny']
    scores = {name: read(path/'metrics.json') for name, path in paths.items()}
    config = {name: read(path/'config.json') for name, path in paths.items()}
    classes = config['transformer']['classes']
    with np.load(ROOT/'data/dataset.npz') as data:
        train_images, train_labels = data['train_images'], data['train_labels']
        test_images, test_labels = data['test_images'], data['test_labels']
    with np.load(ROOT/'data/split_indices.npz') as split:
        train_indices, test_indices = split['train'], split['test']
    predictions = []
    for name in ['classical', 'cnn', 'transformer']:
        csv_path = paths[name]/('test_predictions.csv' if name == 'classical' else 'seed_42/test_predictions.csv')
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        np.testing.assert_array_equal([int(r['original_index']) for r in records], test_indices)
        predictions.append(np.array([int(r['predicted_label']) for r in records]))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5})
    colors = ['#64748b', '#0284c7', '#7c3aed']
    fig, axes = plt.subplots(2, 5, figsize=(6.6, 3.2))
    for label, ax in enumerate(axes.flat):
        index = train_indices[np.flatnonzero(train_labels[train_indices] == label)[0]]
        ax.imshow(train_images[index], interpolation='nearest')
        ax.set_title(classes[label], fontsize=11)
        ax.axis('off')
    fig.tight_layout(pad=.3)
    fig.savefig(ASSETS/'dataset_examples.png', dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    accuracy = [r['test_accuracy']*100 for r in models]
    error = [(r['test_accuracy_seed_std'] or 0)*100 for r in models]
    bars = ax.bar(short, accuracy, yerr=error, capsize=4, color=colors, width=.58)
    for bar, value in zip(bars, accuracy):
        ax.text(bar.get_x()+bar.get_width()/2, value+2, f'{value:.2f}%', ha='center', fontsize=11)
    ax.set(ylabel='Test accuracy, %', ylim=(0, 105))
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSETS/'test_accuracy.png', dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.bar(short, [r['matched_inference_ms_per_image'] for r in models], color=colors, width=.58)
    ax.set(ylabel='ms/image, batch 512', title='CPU SVM; strict-FP32 GPU neural inference')
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ASSETS/'inference_latency.png', dpi=220)
    plt.close(fig)
    per_class = []
    per_class.append([scores['classical']['test']['classification_report'][c]['f1-score'] for c in classes])
    for name in ['cnn', 'transformer']:
        per_class.append([np.mean([v['test']['classification_report'][c]['f1-score'] for v in scores[name]['per_seed'].values()]) for c in classes])
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(10)
    for i, values in enumerate(per_class):
        ax.bar(x+(i-1)*.25, values, width=.25, color=colors[i], label=short[i])
    ax.set_xticks(x, classes, rotation=40, ha='right')
    ax.set(ylabel='Per-class F1', ylim=(0, 1.08))
    ax.legend(ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(.5, 1.13))
    fig.tight_layout()
    fig.savefig(ASSETS/'per_class_f1.png', dpi=220)
    plt.close(fig)
    confusions = []
    for name, title, pred in zip(['svm', 'cnn', 'vit'], short, predictions):
        cm = confusion_matrix(test_labels[test_indices], pred, labels=np.arange(10))
        off = cm.copy()
        np.fill_diagonal(off, 0)
        a, b = np.unravel_index(np.argmax(off), off.shape)
        confusions.append([title, f'{classes[a]} -> {classes[b]}', str(int(off[a, b]))])
        fig, ax = plt.subplots(figsize=(6.6, 5.7))
        ConfusionMatrixDisplay.from_predictions(test_labels[test_indices], pred, labels=np.arange(10),
            display_labels=classes, normalize='true', values_format='.2f', cmap='Blues', colorbar=False,
            xticks_rotation=45, ax=ax)
        ax.set_title(title+': held-out test'+(' / seed 42' if name != 'svm' else ''), fontsize=11)
        fig.tight_layout()
        fig.savefig(ASSETS/f'confusion_{name}.png', dpi=220)
        plt.close(fig)
    wrong = np.flatnonzero(predictions[2] != test_labels[test_indices])
    chosen = np.random.default_rng(42).choice(wrong, min(3, len(wrong)), replace=False)
    example_errors = []
    for j, index in enumerate(chosen):
        file = ASSETS/f'error_{j}.png'
        Image.fromarray(test_images[test_indices[index]]).resize((256, 256), Image.Resampling.NEAREST).save(file)
        example_errors.append({'path': str(file), 'true': classes[int(test_labels[test_indices[index]])],
                               'pred': classes[int(predictions[2][index])]})
    facts = {'models': models, 'short_names': short, 'paths': {k: str(v) for k, v in paths.items()},
        'configs': config, 'scores': scores, 'classes': classes, 'confusions': confusions,
        'example_errors': example_errors, 'split': [45000, 5000, 10000], 'date': '11.11.2026'}
    (OUT/'facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(paths['transformer']/'comparison_all_models.csv', OUT/'Comparison_3models.csv')
    pages = build_pages(facts)
    (OUT/'report_content.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')
    md = ['# Порівняння ML, CNN і vision transformer на CIFAR-10', '\nЛабораторна робота. 11 листопада 2026 року.\n']
    for page in pages:
        md.append('\n## '+page['title']+'\n')
        for block in page['blocks']:
            if block['type'] == 'p':
                md.append(block['text']+'\n')
            elif block['type'] == 'h':
                md.append('\n### '+block['text']+'\n')
            elif block['type'] == 'image':
                md.append(f'![{block.get("caption", "")}](assets/{block["file"]})\n\n'+block.get('caption', '')+'\n')
            elif block['type'] == 'table':
                values = block['rows']
                md.append('| '+' | '.join(values[0])+' |\n| '+' | '.join(['---']*len(values[0]))+' |')
                md.extend('| '+' | '.join(str(c).replace('\n', ' ') for c in row)+' |' for row in values[1:])
                md.append('')
            elif block['type'] == 'ref':
                md.append(block['text']+f' [Першоджерело]({block["url"]})\n')
    (OUT/'Lab_Report_UA.md').write_text('\n'.join(md), encoding='utf-8')
    print('Report evidence and editable content prepared:',OUT)


def P(text): return {'type': 'p', 'text': text}
def H(text): return {'type': 'h', 'text': text}
def T(rows, widths=None): return {'type': 'table', 'rows': rows, 'widths': widths}
def I(file, caption, width=16.7): return {'type': 'image', 'file': file, 'caption': caption, 'width_cm': width}


def build_pages(facts):
    m = facts['models']
    s, c, v = m
    scores, configs = facts['scores'], facts['configs']
    cv = scores['classical']['selected_by_cv']
    quality = [['Модель', 'Test accuracy', 'Test macro-F1']]
    for name, row in zip(facts['short_names'], m):
        acc = fmt(row['test_accuracy']*100)+'%'
        f1 = fmt(row['test_macro_f1'], 4)
        if row['test_accuracy_seed_std'] is not None:
            acc += ' ± '+fmt(row['test_accuracy_seed_std']*100)+' п.п.'
            f1 += ' ± '+fmt(row['test_macro_f1_seed_std'], 4)
        quality.append([name, acc, f1])
    pages = [
        {'title': '1. Постановка задачі та дані', 'blocks': [
            P('Лабораторна робота: експериментальне порівняння класичного ML, CNN та vision transformer. Вхід X - RGB-зображення; ціль y - один із десяти класів CIFAR-10. Мета - виміряти не лише якість, а й локальну ціну її отримання: час, RAM/VRAM, розмір моделі та inference.'),
            P(f'Найвища середня test accuracy у цьому експерименті: {fmt(max(r["test_accuracy"] for r in m)*100)}%. Усі наведені результати отримано фактичним запуском; джерела та checkpoint збережено. Порівняння охоплює різні процедури навчання, включно з попередньо навченим ImageNet backbone.'),
            H('CIFAR-10'),
            P('Повний офіційний набір: 60 000 зображень 32x32, 10 збалансованих класів. Офіційні 50 000 train поділено на 45 000 train та 5 000 validation; офіційні 10 000 test залишено для фінальної оцінки. В усіх моделях індекси split однакові, seed розбиття 42.'),
            T([['Частина', 'Об’єктів', 'Об’єктів на клас'], ['Train', '45 000', '4 500'], ['Validation', '5 000', '500'], ['Test', '10 000', '1 000']], [6, 5, 5.7]),
            I('dataset_examples.png', 'Рис. 1. Перший training-приклад кожного класу; показано справжню низьку роздільність даних.', 14.8),
            P('Архів перевірено за офіційним MD5 c58f30108f718f92721af3b95e74349a. Exact SHA-256 audit не виявив дублікатів, перетину split чи конфліктів міток. Близькі сцени/crops не усуваються цим тестом. На офіційній сторінці не встановлено явної ліцензії dataset; джерело процитовано, дані не включаються до пакета для поширення [7].')]},
        {'title': '2. Огляд сучасних досліджень', 'blocks': [
            P('Відібрано чотири первинні роботи 2024-2025 років. Вони дають контекст якості та ефективності, але не утворюють прямо зіставний leaderboard: відрізняються dataset, split, hardware, pretraining та training recipe.'),
            H('[1] Wu, 2025: small ViT на CIFAR-10'),
            P('Препринт Powerful Design of Small Vision Transformer on CIFAR10 досліджує аугментації, low-rank attention і кілька CLS tokens. Навчання: 100 епох, batch 256, RTX 3090. Автор подає baseline Val/Acc 93,65% і 93,95% для варіанта з двома CLS tokens. Окремий незалежний validation split з опису не встановлено; ці числа не прирівнюємо до нашого test. Корисний висновок - важливість augmentation і розміру представлення.'),
            H('[2] SHViT, CVPR 2024'),
            P('Yun і Ro оптимізують memory access та single-head attention. SHViT-S4 на ImageNet-1K: 79,4% top-1 при 256x256, 16,5 млн параметрів, 14 283 зображень/с на A100, batch 256; 300 епох з нуля. Інші distillation-рецепти мають окремі результати. Для лабораторної беремо потребу вимірювати RAM/VRAM і реальний throughput, а не лише параметри.'),
            H('[3] MobileNetV4, ECCV 2024'),
            P('Qin та співавтори порівнюють CNN/hybrid архітектури на мобільних платформах. На ImageNet-1K MNv4-Conv-S має 73,8% top-1, 3,8 млн параметрів та 2,4 ms на Pixel 6 CPU. Для Hybrid-L стандартний результат 83,4%; 87,0% використовує додаткове pretraining/distillation. Це обґрунтовує явне зазначення пристрою та prior training у нашому порівнянні.'),
            H('[4] TinyNeXt, ICCV 2025'),
            P('Zeng та співавтори поєднують CNN і Lean Single-Head Self-Attention. TinyNeXt-T: 71,5% top-1 на ImageNet-1K, 1,0 млн параметрів, 7257 зображень/с на RTX 3090, inference batch 512. Навчання з нуля: 224x224, 300 епох, 8 RTX 3090. Малий розмір моделі не гарантує однакового latency на різному hardware.'),
            P('Історичні ResNet [5] та DeiT [6] - окремі архітектурні першоджерела, а не заміна цього сучасного огляду. Повні PDF, BibTeX і SHA-256 manifest збережено у sources/.')]},
        {'title': '3. Методологія та ресурси', 'blocks': [
            P('Train використовується для fit параметрів, scaler і навчання нейромереж. Validation - для вибору checkpoint. Test не бере участі в підборі C, аугментацій, epoch budget чи seed. У нейромереж усі три вибори checkpoint зафіксовано до відкриття test-міток.'),
            T([['Аспект', 'Протокол'], ['Classical ML', '3-fold StratifiedKFold тільки всередині train; C та HOG/RGB обрано за CV macro-F1; scaler fit окремо в кожному fold.'], ['Нейромережі', 'Фіксований validation; seeds 42, 123, 2026 замість повної CV через обчислювальну ціну. Test оцінюється для кожного зафіксованого checkpoint.'], ['Агрегування', 'Середнє та population std по трьох neural seeds. Це не CV fold std та не confidence interval. SVM test std не оцінено.'], ['Основні метрики', 'Accuracy, macro Precision/Recall/F1, per-class metrics, confusion matrices. Найкращу епоху визначає validation macro-F1; ties - найраніша епоха.'], ['Hardware', 'Локальний RTX 5060 Ti 16 GiB, близько 31,7 GiB RAM, 16 logical CPU. SVM використовує CPU; neural training - CUDA GPU.'], ['Вимірювання', 'CUDA синхронізується на межах таймінгу. GPU memory - torch allocated/reserved peak; RAM - process-tree RSS кожні 100 ms. Shared pages можуть рахуватися двічі.']], [3.4, 13.3]),
            H('Єдиний inference benchmark'),
            P('Перші 512 тих самих test-зображень, прогрів та п’ять повторів. Відображається median ms/image для batch throughput, а не latency окремого запиту. Включено HOG або resize/normalization, transfer на GPU для neural моделей, forward, argmax та повернення міток. Читання файлів виключено. Обидві neural моделі тут працюють у strict FP32 з TF32 off; SVM - CPU, один native thread.'),
            P('CNN quality збережено з первісного FP32/TF32-allowed запуску; DeiT quality оцінено у strict FP32. Ця числова відмінність і різні precision/hardware режими зазначаються, а не маскуються. Це практичне порівняння процедур, не контрольований експеримент лише над attention.')]},
        {'title': '4. Архітектури та навчання', 'blocks': [
            H('4.1. HOG + RGB + Linear SVM'),
            P('З кожного зображення отримано 1764 HOG ознаки (9 орієнтацій, cell 4x4, block 2x2, L2-Hys) та 96 RGB bins. StandardScaler входить у sklearn pipeline. Порівняно HOG і HOG+RGB та C = 0.0001, 0.001, 0.01: 18 CV fits і один refit. Обрано HOG+RGB та C=0.001. Фінальний fit - лише на спільному train, validation збережено.'),
            P(f'Найкращий CV macro-F1: {fmt(cv["cv_f1_macro_mean"],4)} ± {fmt(cv["cv_f1_macro_std"],4)}; коефіцієнтів та intercepts SVM: {s["parameters_or_svm_coefficients"]:,}. Цей count не включає сам алгоритм HOG та fitted scaler.'),
            H('4.2. ResNet-18 з нуля'),
            P('Torchvision ResNet-18: stem 3x3 stride 1 замість 7x7 stride 2; maxpool прибрано; вихід 10 класів. Немає pretrained weights. RGB mean/std обчислено лише на train. Train augmentation: випадковий reflect-padded crop з padding 4 і horizontal flip 0.5. 60 епох, batch 256, SGD Nesterov, LR 0.1, momentum 0.9, weight decay 5e-4, cosine decay. BF16 autocast training, FP32 weights; evaluation FP32 з дозволеним TF32.'),
            H('4.3. DeiT-Tiny з ImageNet pretraining'),
            P('Повний pretrained backbone deit_tiny_patch16_224.fb_in1k завантажено з незмінної Hugging Face revision. Лише head замінено на 10 класів; усі параметри fine-tune. CIFAR augmentation виконується на 32x32, потім bicubic resize до 224x224 та published ImageNet normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225). Upsampling не додає деталей.'),
            P(f'{configs["transformer"]["epochs"]} епох, batch {configs["transformer"]["batch_size"]}, AdamW: backbone LR 5e-5, head LR 5e-4, weight decay 0.05; один epoch warmup, потім cosine; clip norm 1.0. BF16 training, strict FP32 evaluation. Локально не вимірювалася і не включена початкова вартість ImageNet pretraining. Кількість параметрів: {v["parameters_or_svm_coefficients"]:,} проти {c["parameters_or_svm_coefficients"]:,} у CNN.')]},
    ]
    seeds = [['Модель / seed', 'Вибрана епоха', 'Accuracy, %', 'Macro-F1']]
    for name, title in [('cnn', 'ResNet'), ('transformer', 'DeiT')]:
        for seed, result in scores[name]['per_seed'].items():
            seeds.append([title+' / '+seed, str(result['best_epoch']), fmt(result['test']['accuracy']*100), fmt(result['test']['f1_macro'], 4)])
    cost_rows = [['Модель', 'Fit / seed, хв', 'Усі запуски, хв', 'Модель, MiB', 'Inference, ms/зобр.']]
    memory_rows = [['Модель', 'Peak RAM, GiB', 'Peak GPU allocated, GiB', 'Параметри / coefficients']]
    for name, row in zip(facts['short_names'], m):
        cost_rows.append([name, fmt(row['local_train_seconds_per_fit']/60),
            fmt(row['total_experiment_training_seconds']/60), fmt(row['checkpoint_bytes']/2**20), fmt(row['matched_inference_ms_per_image'], 3)])
        memory_rows.append([name, fmt(row['peak_training_ram_mib']/1024),
            '-' if row['peak_cuda_allocated_mib'] == 0 else fmt(row['peak_cuda_allocated_mib']/1024),
            f'{row["parameters_or_svm_coefficients"]:,}'.replace(',', ' ')])
    gain_vs_svm = (v['test_accuracy']-s['test_accuracy'])*100
    gain_vs_cnn = (v['test_accuracy']-c['test_accuracy'])*100
    inference_ratio = v['matched_inference_ms_per_image']/c['matched_inference_ms_per_image']
    quality_winner = facts['short_names'][int(np.argmax([r['test_accuracy'] for r in m]))]
    pages += [
        {'title': '5. Результати якості', 'blocks': [
            T(quality, [4.3, 6.2, 6.2]),
            P('У нейромереж наведено середнє ± std по трьох незалежних training seeds. SVM має один фінальний test-результат; його CV std не перенесено на test. Це різні оцінки нестабільності.'),
            I('test_accuracy.png', 'Рис. 2. Accuracy на тому самому test. Error bars нейромереж - std по seeds, не confidence intervals.'),
            T(seeds, [4.9, 3.8, 4., 4.]),
            P('Epoch selection перевірено за історіями: checkpoint відповідає найранішому максимуму validation macro-F1. Усі три selections у кожному neural експерименті зафіксовано до test. Жоден seed не обрано як найкращий за test.')]},
        {'title': '6. Якість проти обчислювальної ціни', 'blocks': [
            T(cost_rows, [3.6, 3.1, 3.3, 3.1, 3.6]),
            P('Fit SVM - тільки фінальний refit; усі запуски SVM включають 18 CV fits і refit. Neural fit/seed включає validation і checkpoint I/O; усі запуски - сума трьох seeds. Для DeiT не включено початкове ImageNet pretraining. Це локальна ціна адаптації, не повна ціна створення representation.'),
            T(memory_rows, [3.6, 3.7, 4.7, 4.7]),
            P('RAM SVM охоплює CV worker processes і може двічі рахувати shared pages. GPU peaks нейромереж включають resident images та startup/autotuning; у таблиці максимум по seeds, а не середнє. Параметри SVM - coefficients/intercepts, без HOG-коду і scaler. Серіалізовані розміри є фактичними checkpoint sizes.'),
            I('inference_latency.png', 'Рис. 3. Єдиний batch-512 benchmark; обидві neural моделі strict FP32 на GPU, SVM на CPU. ms/image не є single-request latency.'),
            P(f'DeiT відносно CNN: {fmt(gain_vs_cnn)} п.п. accuracy; відносно SVM: {fmt(gain_vs_svm)} п.п. У цьому запуску DeiT inference займає приблизно {fmt(inference_ratio,1)}x часу CNN. Не можна приписати різницю лише attention: є ImageNet prior training, 224x224 проти 32x32 та різні optimizer/epoch budgets.')]},
        {'title': '7. Аналіз помилок', 'blocks': [
            I('per_class_f1.png', 'Рис. 4. SVM per-class F1; для neural моделей - середнє per-class F1 по трьох seeds. Це не ensemble.'),
            T([['Модель', 'Найчастіша off-diagonal помилка', 'Кількість']] + facts['confusions'], [4, 8.7, 4]),
            P('Confusion table та матриці у додатках використовують наперед обраний seed 42 для neural моделей. Кожен істинний клас має 1000 test-прикладів, тому counts можна інтерпретувати в межах цього support. Найбільші off-diagonal клітинки знайдено після фіксації моделей і не використано для донавчання.'),
            P('HOG і глобальні кольорові гістограми описують форму/текстуру, але слабше відділяють семантично схожі об’єкти. CNN навчає локальні ознаки, а DeiT використовує попередньо навчене представлення. Це узгоджується з per-class результатами, але не ізолює причинний внесок окремих компонентів.'),
            P('Частина помилок пов’язана з неоднозначними tiny-зображеннями, малим об’єктом, фоном і схожими класами. На 32x32 важливі деталі можуть бути втрачені ще у вихідних даних; resize до 224x224 не відновлює їх. Випадкові приклади помилок і повні predictions CSV збережено у result folders.')]},
        {'title': '8. Висновки, обмеження та відтворення', 'blocks': [
            P(f'Найвища середня test accuracy серед перевірених процедур - {quality_winner}. Вибір для використання залежить від вимог: SVM є компактним CPU baseline; CNN дає високу якість та коротший measured GPU inference; transformer перевіряє, чи виправдовує ImageNet transfer додатковий inference cost. Значення для цього рішення наведено у таблицях, а не виведено лише з типу архітектури.'),
            H('Обмеження'),
            P('1. Різні prior training, resolutions і learning budgets не дозволяють зробити чистий висновок «attention кращий за convolutions». Вартість ImageNet pretraining невідома в цьому експерименті.'),
            P('2. SVM використовує CPU, нейромережі GPU; це не вимірювання speedup одного алгоритму. Пам’ять RAM та VRAM - окремі величини; sampling RSS має обмеження. Енергія і грошова вартість не вимірювалися.'),
            P('3. Лише три neural seeds, один dataset, контрольована 10-class closed-set задача. Огляд джерел обмежений чотирма сучасними роботами; exact hashes не виявляють near-duplicates або можливий overlap із pretraining corpus.'),
            P('4. У первісному CNN GPU reload 29 999 із 30 000 labels збіглися точно, одна top-2 near-tie змінилася; accuracy не змінилася. Original metrics і weights збережено. Повний numerical audit описує tolerance <=0.001 logits. Transformer перевірено окремо з strict FP32; точний reload підтверджено на тому самому GPU/software.'),
            H('Відтворення та збережені матеріали'),
            P('Робоча папка: E:/labs/SHI_LABA_BIG. Скрипти: prepare_data.py, train_baseline.py, train_cnn.py, train_transformer.py. Залежності кожного етапу зафіксовані у requirements-*-lock.txt; базовий SVM lock також збережено.'),
            P('Результати: results/latest_classical.txt, latest_cnn.txt, latest_transformer.txt. Там є config, split/model hashes, histories, best/last checkpoint, predictions, per-class metrics, timing і memory. Перевірка: verify_results.py, verify_cnn.py, verify_transformer.py; unit/regression checks: python -m unittest discover -v.'),
            P('Навчання/аналіз не потребують notebook. PDF цього звіту, редагований Markdown, CSV порівняння та шість слайдів зі speaker notes є окремими deliverables. Джерела PDF і BibTeX залишаються у sources/.')]},
    ]
    refs = [
        ('[1] Gent Wu. Powerful Design of Small Vision Transformer on CIFAR10. arXiv preprint, 2025; 2501.06220v1.', 'https://arxiv.org/abs/2501.06220'),
        ('[2] Seokju Yun, Youngmin Ro. SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design. CVPR 2024; arXiv 2401.16456v2.', 'https://arxiv.org/abs/2401.16456'),
        ('[3] Danfeng Qin et al. MobileNetV4: Universal Models for the Mobile Ecosystem. ECCV 2024; arXiv 2404.10518v2.', 'https://arxiv.org/abs/2404.10518'),
        ('[4] Fanhong Zeng et al. An Efficient Hybrid Vision Transformer for TinyML Applications. ICCV 2025 (TinyNeXt).', 'https://www.openaccess.thecvf.com/content/ICCV2025/papers/Zeng_An_Efficient_Hybrid_Vision_Transformer_for_TinyML_Applications_ICCV_2025_paper.pdf'),
        ('[5] Kaiming He et al. Deep Residual Learning for Image Recognition. CVPR 2016; arXiv 1512.03385.', 'https://arxiv.org/abs/1512.03385'),
        ('[6] Hugo Touvron et al. Training data-efficient image transformers and distillation through attention. ICML 2021; arXiv 2012.12877.', 'https://arxiv.org/abs/2012.12877'),
        ('[7] Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images, 2009. Офіційна CIFAR-10 download page та checksum.', 'https://www.cs.toronto.edu/~kriz/cifar.html'),
        ('[8] Official timm pretrained model card: deit_tiny_patch16_224.fb_in1k. ImageNet-1K, Apache-2.0; локально pinned revision і SHA-256.', 'https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k'),
        ('[9] PyTorch 2.11 numerical accuracy documentation: floating-point reproducibility і TF32.', 'https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html'),
    ]
    pages.append({'title': '9. Джерела', 'blocks': [P('Перші чотири роботи - сучасний related work; ResNet і DeiT - додаткові архітектурні джерела. Числа авторів у розділі 2 не змішуються з власними результатами у розділах 5-7.')]
        + [{'type': 'ref', 'text': text, 'url': url} for text, url in refs]
        + [P('У sources/ збережено шість research PDF, RELATED_WORK_UA.md, references.bib та manifest.json з URL, датами, версіями, розмірами, кількістю сторінок і SHA-256. Upstream weights/config/model card і окрема provenance - у data/pretrained/deit_tiny/. Усі власні таблиці та графіки побудовано із зафіксованих experiment artifacts.')]})
    for name, title in zip(['svm', 'cnn', 'vit'], facts['short_names']):
        pages.append({'title': 'Додаток. Confusion matrix: '+title, 'blocks': [
            I(f'confusion_{name}.png', 'Нормування за істинним класом, однакові 10 000 test-прикладів. Для neural моделей використано наперед обраний seed 42.'),
            P('Рядки - істинні класи, стовпці - прогноз. Кожен клас має support 1000. Значення округлено до двох знаків; сума округлених клітинок може трохи відрізнятися від 1. Повні integer counts можна відновити з test_predictions.csv.')]})
    return pages


if __name__ == '__main__':
    main()
