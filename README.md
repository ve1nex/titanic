# Titanic

Бинарная классификация для [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic): прогноз выживания пассажира по табличным признакам. Target — `Survived`, метрика — accuracy. Исходные данные: 891 обучающий и 418 тестовых объектов.

Проект объединяет Classic ML, MLP и ансамбль RandomForest + XGBoost + MLP. История экспериментов и результаты ведутся отдельно в заметке Obsidian; текущие метрики также сохраняются в файлы проекта.

## Установка

Python 3.12. Все команды выполняются из корня проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

В Windows активация: `.venv\Scripts\activate`.

Файлы `train.csv` и `test.csv` должны находиться в корне проекта.

## Полное обучение

Для обучения с заданными гиперпараметрами установите `tuning.enabled = False` в `classic/config.py` и `dl/config.py`.

```bash
python main.py --mode train --pipeline final
```

Команда загружает данные, обучает RandomForest, XGBoost и MLP с кросс-валидацией по пяти фолдам, сохраняет модели и OOF-предсказания, затем собирает ансамбль и сабмит.

Имена новых экспериментов формируются из `new_experiment_prefix` в корневом `config.py`:

```python
"new_experiment_prefix": "clean_cv_v2",
```

В этом примере создаются `clean_cv_v2_rf`, `clean_cv_v2_xgb`, `clean_cv_v2_dl`. Перед повторным обучением выбирайте свободный префикс: существующие эксперименты защищены от перезаписи.

## Отчёт и сабмит из сохранённых результатов

Для уже выполненного запуска `clean_cv_v1` в корневом `config.py` укажите:

```python
"mode": "report",
"pipeline": "final",
"ensemble": {
    "rf_experiment": "clean_cv_v1_rf",
    "xgb_experiment": "clean_cv_v1_xgb",
    "dl_experiment": "clean_cv_v1_dl",
    "voting": "soft",
    "weights": [1, 1, 1],
    "threshold": 0.5,
},
```

```bash
python main.py
```

Или явно:

```bash
python main.py --mode report --pipeline final
```

Режим `report` использует сохранённые OOF и тестовые предсказания, рассчитывает метрики ансамбля и создаёт:

| Файл | Назначение |
|---|---|
| `outputs/submission.csv` | Сабмит Kaggle: `PassengerId`, `Survived` |
| `outputs/results.csv` | Таблица CV сохранённых моделей и ансамбля |
| `outputs/metrics.json` | Метрики текущего ансамбля |

После обучения с новым префиксом обновите три имени в секции `ensemble`, чтобы последующий отчёт использовал новые эксперименты.

Для пересчёта тестовых предсказаний из сохранённых весов:

```bash
python main.py --mode inference --pipeline final
```

Для отдельных экспериментов доступны собственные точки входа:

```bash
python classic/main.py general.mode=train general.experiment_name=logreg_new model.name=LogisticRegression
python dl/main.py general.mode=train general.experiment_name=mlp_new
```

## Конфиги

| Конфиг | Что настраивает |
|---|---|
| `config.py` в корне | Режим запуска, выбор пайплайна, папка результатов, имена экспериментов, voting и веса ансамбля |
| `classic/config.py` | Признаки, заполнение пропусков, кодирование категорий, модель, гиперпараметры, CV и Optuna |
| `dl/config.py` | Архитектура MLP, batch size, число эпох, optimizer, scheduler, loss, CV и Optuna |

Корневой конфиг связывает готовые пайплайны: задаёт, что запускать и какие сохранённые эксперименты объединять. Подробные параметры обучения остаются в соответствующих конфигах Classic и DL. Аргументы `--mode` и `--pipeline` переопределяют корневые настройки для конкретного запуска.

В `classic/config.py` и `dl/config.py` пути должны вычисляться так:

```python
PROJECT_ROOT = Path(__file__).resolve().parent

"path_to_train_dataset": str(Path(__file__).resolve().parents[1] / "train.csv"),
"path_to_test_dataset": str(Path(__file__).resolve().parents[1] / "test.csv"),
"path_to_checkpoints": str(PROJECT_ROOT / "checkpoints/${general.experiment_name}"),
```

Отчёт ожидает Classic-артефакты в `classic/checkpoints/`, DL-артефакты — в `dl/checkpoints/`.

## Подготовка данных и модели

Classic ML использует заполнение пропусков, масштабирование числовых признаков и OneHotEncoder для категорий. Из `Name` извлекается `Title`; `Ticket` и `Cabin` исключаются. Доступны LogisticRegression, KNN, RandomForest и XGBoost.

DL использует MLP со скрытыми слоями `[20, 28]`, ReLU, AdamW, CrossEntropyLoss и ReduceLROnPlateau. Предобработка обучается на тренировочной части каждого фолда. Препроцессор MLP сохраняется рядом с весами соответствующего фолда.

Валидация — StratifiedKFold, пять фолдов, `shuffle=True`, seed `1027309`. Итоговый soft voting усредняет вероятности RandomForest, XGBoost и MLP с одинаковыми весами; порог классификации — `0.5`.

## Структура проекта

| Файл / каталог | Назначение |
|---|---|
| `main.py`, `config.py` | Общий запуск и выбор экспериментов |
| `requirements.txt` | Зависимости |
| `train.csv`, `test.csv` | Исходные данные |
| `EDA.ipynb` | Анализ пропусков, распределений, выбросов и связи признаков с выживаемостью |
| `classic/` | Classic ML: данные, признаки, модели, обучение, CV и предсказания |
| `dl/` | MLP: предобработка, обучение, CV и предсказания |
| `ensemble/ensemble_soft.py` | Объединение сохранённых предсказаний |
| `classic/checkpoints/`, `dl/checkpoints/` | Модели, OOF, тестовые предсказания, конфиги и метрики |
| `outputs/` | Сабмит и отчёт текущего решения |
