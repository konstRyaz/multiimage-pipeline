# Воспроизводимая оценка базового конвейера

Команда `src/baseline_evaluate.py` реализует пять раздельных стадий:

```text
prepare → calibrate → validate → freeze → evaluate
```

Оценка не связана с экспериментами фильтрации из `src/run_experiment.py`, но
использует тот же пакет `face_pipeline`, модель InsightFace и стиль каталогов
результатов. Пример `configs/baseline_evaluation_v1.json` уже настроен на:

```text
/mnt/storage-1/k.ryazanov/datasets/baseline_v1
```

Пути можно менять в конфигурации; в коде они не зашиты.

## Установка и предварительная проверка

```bash
cd /home/k.ryazanov/multiimage-pipeline
source .venv/bin/activate
python -m pip install -r requirements-baseline.txt

python src/baseline_evaluate.py preflight \
  --config configs/baseline_evaluation_v1.json \
  --run-dir runs/baseline_evaluation_v1
```

Проверяются корень данных, каталог результатов, свободное место, зависимости,
устройство и доступность CUDA-провайдера. Структура и полнота разметки
проверяются стадией `prepare`.

Для короткой проверки интерфейса на ограниченном числе изображений передайте
одинаковый `--limit N` каждой стадии. Такой запуск имеет отдельный отпечаток и
его замороженные параметры несовместимы с полным запуском.

## Полный запуск

```bash
CONFIG=configs/baseline_evaluation_v1.json
RUN=runs/baseline_evaluation_v1

python src/baseline_evaluate.py prepare   --config "$CONFIG" --run-dir "$RUN"
python src/baseline_evaluate.py calibrate --config "$CONFIG" --run-dir "$RUN"
python src/baseline_evaluate.py validate  --config "$CONFIG" --run-dir "$RUN"
python src/baseline_evaluate.py freeze    --config "$CONFIG" --run-dir "$RUN"
python src/baseline_evaluate.py evaluate  --config "$CONFIG" --run-dir "$RUN" \
  --frozen "$RUN/frozen_parameters.json"
```

Для долгого запуска на одной GPU рекомендуется один сеанс `tmux` и единый
журнал оболочки. Параметр `--progress-interval` не входит в отпечаток
эксперимента и не влияет на кэш или метрики:

```bash
scripts/run_baseline_full_1gpu.sh \
  --run-dir /mnt/storage-1/k.ryazanov/runs/baseline_evaluation_full_1gpu_v1 \
  --gpu 3 \
  --progress-interval 30
```

Сценарий выполняет предварительную проверку, затем все пять стадий. При
повторном запуске он пропускает стадии с корректным состоянием `complete` и
продолжает незавершённую стадию через поэлементный кэш. Индекс `--gpu` задаёт
физическую карту; внутри ограниченного `CUDA_VISIBLE_DEVICES` конфигурация
продолжает корректно использовать `model.device=0`.

Эквивалентная последовательность команд сценария:

```bash
CONFIG=configs/baseline_evaluation_v1.json
RUN=/mnt/storage-1/k.ryazanov/runs/baseline_evaluation_full_1gpu_v1
PROGRESS_INTERVAL=30

set -o pipefail
mkdir -p "$RUN"

run_stage() {
  stage="$1"
  shift
  echo "=== $(date --iso-8601=seconds) START $stage ===" | tee -a "$RUN/full_run.log"
  python src/baseline_evaluate.py "$stage" \
    --config "$CONFIG" --run-dir "$RUN" \
    --progress-interval "$PROGRESS_INTERVAL" "$@" \
    2>&1 | tee -a "$RUN/full_run.log"
  status=${PIPESTATUS[0]}
  echo "=== $(date --iso-8601=seconds) END $stage status=$status ===" | tee -a "$RUN/full_run.log"
  return "$status"
}

python src/baseline_evaluate.py preflight --config "$CONFIG" --run-dir "$RUN" &&
run_stage prepare &&
run_stage calibrate &&
run_stage validate &&
run_stage freeze &&
run_stage evaluate --frozen "$RUN/frozen_parameters.json"
```

Цепочка на `&&` принципиальна: следующая стадия не начнётся после ошибки.
Повторный запуск той же стадии безопасно использует корректные файлы кэша.

## Живой прогресс, ETA и промежуточные метрики

Во время обработки каждые 30 секунд в терминал и `full_run.log` выводятся:

- текущая стадия и часть датасета;
- обработанное и полное число элементов, процент;
- средняя скорость с начала текущей части;
- динамическая оценка оставшегося времени;
- число новых вычислений, попаданий в кэш и ошибок;
- техническое покрытие: наличие детекций, главного лица CelebA или пригодного
  эмбеддинга XQLFW.

В другом SSH-окне состояние можно смотреть без подключения к `tmux`:

```bash
cd /home/k.ryazanov/multiimage-pipeline
source .venv/bin/activate
RUN=/mnt/storage-1/k.ryazanov/runs/baseline_evaluation_full_1gpu_v1

watch -n 15 "python src/baseline_evaluate.py status --run-dir '$RUN' --metrics 3"
```

Для непрерывного текстового журнала:

```bash
tail -F "$RUN/full_run.log"
```

Файлы наблюдения:

```text
RUN/progress/current.json                 # последний атомарный снимок и ETA
RUN/progress/events.jsonl                 # полная хронология прогресса
RUN/progress/intermediate_metrics.jsonl   # содержательные контрольные метрики
```

Промежуточные содержательные показатели записываются сразу после готовности
каждого блока: WIDER train, CelebA train, XQLFW scores, CelebA val,
десятиблочная проверка XQLFW, замороженные параметры и три итоговых блока.
Техническое покрытие на незавершённой части не является оценкой качества на
полном датасете. Итоговыми считаются только значения завершённой стадии.

`evaluate` требует явный `--frozen`. Она проверяет версию схемы, отпечаток
эксперимента и наличие отдельных порогов детекции WIDER FACE, верификации
XQLFW и кластеризации CelebA. В этой функции нет вызовов подбора порога.

## Какие данные использует каждая стадия

| Стадия | WIDER FACE | XQLFW | CelebA |
| --- | --- | --- | --- |
| `prepare` | проверка `train` и `val` | проверка 6000 пар и изображений | проверка всех официальных файлов и непересечения личностей |
| `calibrate` | рабочий порог на всём `train` | извлечение и кэширование 6000 оценок пар | показатели кандидатов на `train` |
| `validate` | не используется | 10 запусков: 9 блоков для порога, 1 для проверки | сравнение кандидатов на `val` |
| `freeze` | сохраняется порог с `train` | медиана десяти порогов | числовой порог повторно подбирается на `train + val` |
| `evaluate` | один прогон на всём `val` | применение замороженного рабочего порога | один прогон на `test` |

Официальный WIDER `test` не используется. CelebA `test` не читается стадиями
подбора. Порог XQLFW хранится как `xqlfw.verification_threshold`, а порог
CelebA — как `celeba.clustering_threshold`; это разные параметры.

## Протоколы и метрики

### WIDER FACE

Детектор создаётся только с модулем `detection`. Поэтому рамка остаётся в
оценке независимо от наличия ключевых точек или эмбеддинга. На `train`
сохраняется кривая `precision/recall/F1`, выбирается максимум F1 и, если он
существует, дополнительная рабочая точка с `precision ≥ 0,99`.

AP Easy/Medium/Hard рассчитывается по официальным MAT-спискам сложности,
глобальной нормализации оценок, правилам игнорирования и 1000 рабочим точкам,
совместимым с `widerface_evaluate`. В отчёте используется формулировка
«строго совместимый с официальным протоколом», а не утверждение о вызове
оригинального MATLAB/Python-сценария авторов.

### XQLFW

Файл должен иметь заголовок `10 300` и ровно 6000 последующих строк. Каждый
последовательный блок содержит 300 положительных и 300 отрицательных пар.
В каждом из десяти запусков порог максимальной точности определяется только на
5400 парах других девяти блоков и применяется к 600 парам оставшегося блока.
Сохраняются все пороги, результаты блоков, среднее и стандартное отклонение
точности, ROC AUC, EER и TAR при нескольких FAR. Если ожидается менее десяти
ложных принятий, результат TAR сопровождается предупреждением.

### CelebA

Главное лицо — детекция с максимальным IoU с официальной рамкой; требуется
IoU не ниже `main_face_min_iou`. Отсутствующее главное лицо учитывается
отдельно. Для полных частей используется HNSW-индекс FAISS, поиск ограниченного
числа соседей и компоненты порогового графа. Это масштабируемая приближённая
кластеризация: она не создаёт матрицу `N × N`, но может не увидеть ребро, не
попавшее в список ближайших соседей. Число соседей фиксируется в конфигурации и
замороженном файле. Для фикстур до 10 тысяч записей используется точный
блочный расчёт.

Парные метрики вычисляются по таблице сопряжённости, без перебора пар.
Сохраняются pairwise precision/recall/F1, B-cubed, ARI, число кластеров, доли
ошибочно объединённых и раздробленных личностей, покрытие, а также отдельные
срезы по всем личностям и по личностям минимум с двумя изображениями.
Одиночные личности остаются в данных и не входят только в знаменатель доли
раздробленных личностей.

## Возобновление и результаты

Каждое изображение кэшируется отдельным атомарно записанным NPZ-файлом с
размером, временем изменения и отпечатком модели/эксперимента. После прерывания
повторите ту же команду: корректные элементы будут прочитаны из кэша. Ошибки
отдельных изображений сохраняются как предупреждения. XQLFW требует обе
стороны всех пар и останавливается при пропуске, поскольку иначе официальный
протокол из 6000 пар нарушается.

```text
RUN/
├── resolved_config.json
├── run_metadata.json
├── dataset_summary.json
├── frozen_parameters.json
├── stages/
│   ├── prepare.json
│   ├── calibrate.json
│   ├── validate.json
│   ├── freeze.json
│   └── evaluate.json
├── cache/{wider,xqlfw,celeba}/
├── progress/
│   ├── current.json
│   ├── events.jsonl
│   └── intermediate_metrics.jsonl
├── calibration/
├── validation/summary.json
└── evaluation/
    ├── metrics.json
    ├── warnings.json
    └── REPORT.md
```

`run_metadata.json` фиксирует исходный Git-коммит, полную конфигурацию,
начальное значение генератора, устройство, модель, существенную предобработку
и версии пакетов. Тяжёлые кэши и результаты игнорируются Git и не входят в
архив исходного кода.

## Проверки на целевой машине

После модульных тестов выполните полный `preflight`, затем пять стадий выше.
Проверьте фактические количества в `dataset_summary.json`, наличие всех трёх
AP в `evaluation/metrics.json`, покрытие каждого набора и список пропусков в
`evaluation/warnings.json`. Для подтверждения воспроизводимости повторите
`evaluate` с тем же замороженным файлом: дорогие признаки должны прийти из
кэша, а значения метрик — совпасть.
