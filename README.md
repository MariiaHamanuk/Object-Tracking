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

<!-- TODO: таблиця з results/ -->
