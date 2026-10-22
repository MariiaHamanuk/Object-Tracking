# Object-Tracking

## Мета

<!-- TODO: коротко описати задачу: покращення трекінгу (OSTrack) під час оклюзій за допомогою ембедингів V-JEPA -->

## Структура

```
Object-Tracking/
├── README.md              # мета, як запустити, результати
├── environment/           # окремі середовища для OSTrack і V-JEPA
├── baselines/
│   └── OSTrack/           # git submodule (форк botaoye/OSTrack)
├── configs/               # yaml: baseline, fusion_v1, fusion_v2
├── scripts/               # точки входу (запуск експериментів, кешування, оцінка)
├── src/
│   ├── data/              # парсинг LaSOT, вибір категорій, split, підвибірки
│   ├── eda/               # скрипти й ноутбуки для EDA
│   ├── vjepa/             # завантаження енкодера, кешування ембедингів
│   ├── tracker/           # обгортка над OSTrack і наші зміни
│   └── eval/              # метрики, occlusion recovery rate, розбивка за оклюзіями
├── notebooks/             # лише для досліджень, не для основної логіки
├── results/               # маленькі csv/json з метриками (комітяться)
└── docs/                  # proposal, нотатки, постер
```

## Як запустити

```bash
# 1. Клонувати разом із сабмодулями
git clone --recurse-submodules <repo-url>
# або, якщо вже склоновано:
git submodule update --init --recursive

# 2. Середовища
conda env create -f environment/ostrack.yml
conda env create -f environment/vjepa.yml
```

### EDA

EDA не потребує кадрів, тож 230 ГБ качати не треба. Скрипт читає zip-архіви з HuggingFace частинами
і забирає лише анотації, кілька кадрів на відео, кількість кадрів, split і офіційні атрибути
(~300 МБ на весь LaSOT, ~20–30 хв):

```bash
python scripts/fetch_lasot_annotations.py        # усі 70 класів -> ./data (або $LASOT_ROOT)
# потім відкрити src/eda/eda_lasot.ipynb
```

Повторний запуск пропускає вже завантажені класи.

<!-- TODO: команди для запуску baseline / fusion -->

## Результати

Оцінка: `python scripts/evaluate.py --results <папка з <seq>.txt> --name <експеримент>`
(протокол OSTrack/pytracking; ORR — частка епізодів зникнення цілі, після яких трекер знову на цілі
(IoU > 0.5) протягом 30 кадрів, серед епізодів, де він був на цілі перед зникненням).

| Метод | Вибірка | AUC | P | NP | ORR | ORR, оклюзія 31–150 кадрів |
|---|---|---|---|---|---|---|
| OSTrack-256 (результати авторів) | test, 280 | 69.11 | 75.22 | 78.68 | 72.3 | 50.6 |
| OSTrack-256 (результати авторів) | dev_occlusion, 40 | 56.51 | 63.78 | 64.69 | 66.5 | 49.0 |

`configs/splits/dev_occlusion.txt` — 40 test-відео з найбільшою щільністю оклюзій (по одному на клас),
для розробки: `python scripts/make_dev_subset.py`.
