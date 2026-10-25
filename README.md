# Object-Tracking

Трекінг одного об'єкта на відео з оклюзіями: OSTrack, доповнений замороженою відеомоделлю V-JEPA 2
(енкодер і предиктор). Дані: LaSOT.

## Структура

```
baselines/OSTrack/   OSTrack (git submodule)
configs/             yaml і списки відео (configs/splits)
scripts/             запуск трекінгу, кешування V-JEPA, навчання, оцінка
src/data/            LaSOT: читання анотацій, потокове завантаження з HuggingFace, підвибірки
src/eda/             статистика й графіки для EDA
src/vjepa/           енкодер і предиктор V-JEPA 2
src/tracker/         обгортка над OSTrack, релокалізація, fusion, детектор
src/eval/            AUC / P / NP, ORR
notebooks/           ноутбуки для Kaggle
results/             метрики (csv/json)
docs/                proposal, опис інтеграції
```

## Запуск

```bash
git clone --recurse-submodules <repo-url>
conda env create -f environment/ostrack.yml
conda env create -f environment/vjepa.yml
```

Експерименти запускаються на Kaggle (GPU T4 x2, Internet on): ноутбук клонує репозиторій, кадри LaSOT
читаються з HuggingFace потоково, тому датасет підключати не треба. Основні ноутбуки:
`notebooks/kaggle_final.ipynb` (фінальна система та абляції на dev) і
`notebooks/kaggle_best_holdout.ipynb` (перевірка на 240 test-відео поза dev).

Окремі кроки локально:

```bash
python scripts/fetch_lasot_annotations.py                 # анотації для EDA і оцінки
python scripts/cache_vjepa.py --sequences configs/splits/dev_occlusion.txt --surprise
python scripts/run_ostrack_lasot.py --checkpoint OSTrack_ep0300.pth.tar \
    --sequences configs/splits/dev_occlusion.txt --vjepa-cache cache/vjepa --reloc-verify 3 --verify-min 0.6
python scripts/evaluate.py --results results_raw/<run>/lasot --name <run> --sequences configs/splits/dev_occlusion.txt
```

## Набори відео

- `configs/splits/dev_occlusion.txt`: 40 test-відео з найбільшою кількістю оклюзій, по одному на клас; на
  них порівнюються всі варіанти.
- `configs/splits/test_holdout.txt`: решта 240 test-відео, лише для фінальної перевірки.
- `configs/splits/train_fusion.txt`: 280 train-відео (4 на клас) для навчання fusion.
- `configs/splits/warning_train.txt`: 40 train-відео для навченої тривоги про оклюзію.

## Метрики

AUC, Precision (20 px) і Normalized Precision (0.2) за протоколом pytracking. ORR_all: частка всіх епізодів
оклюзії, після яких трекер протягом 30 кадрів знову на цілі (IoU > 0.5). ORR: те саме лише для епізодів,
де трекер був на цілі перед оклюзією.
