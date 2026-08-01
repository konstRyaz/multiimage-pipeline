# Наборы данных для `baseline_v1`

Установщик готовит WIDER FACE train и validation, исходный невыравненный XQLFW и,
после явного принятия условий, CelebA In-The-Wild. Он не использует `sudo`,
`apt-get`, системный Python и каталоги вне `DATA_ROOT`.

## Быстрый запуск

```bash
cd /путь/к/multiimage-pipeline
export DATA_ROOT="/путь/с/достаточным/местом/datasets"

bash scripts/download_baseline_datasets.sh
bash scripts/verify_baseline_datasets.sh
```

Первая команда автоматически готовит WIDER FACE и XQLFW. CelebA без
подтверждения условий пропускается.

## CelebA

Официальные условия разрешают набор только для некоммерческих исследований,
запрещают перераспространение, а разметку идентичностей авторы выдают по
запросу. После принятия условий запустите:

```bash
bash scripts/download_baseline_datasets.sh \
  --accept-celeba \
  --require-celeba
```

Если Google Drive не разрешает автоматическую загрузку, скачайте файлы через
официальную страницу и передайте каталог с ними:

```bash
bash scripts/download_baseline_datasets.sh \
  --accept-celeba \
  --celeba-source /путь/к/скачанным_файлам_CelebA \
  --require-celeba
```

Полученный от авторов `identity_CelebA.txt` можно положить непосредственно в
`$DATA_ROOT/celeba/annotations/`, затем повторить строгую проверку:

```bash
bash scripts/verify_baseline_datasets.sh --require-celeba
```

Условия и загрузка: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Что проверяется

- WIDER FACE: 12880 train и 3226 validation изображений, разметка рамок и файлы подмножеств
  Easy/Medium/Hard;
- XQLFW: 7263 исходных изображения, протокол пар и оценки качества;
- CelebA: 202599 изображений и разметки атрибутов, идентичностей, рамок,
  ориентиров и официального разбиения.

Локальные SHA-256 скачанных архивов записываются рядом с ними. Это фиксация
полученных байтов для воспроизводимости, а не подмена отсутствующих у авторов
официальных контрольных сумм.

Полный контур метрик описан в [`baseline_evaluation.md`](baseline_evaluation.md).
Для него используется `src/baseline_evaluate.py`; команда экспериментов
фильтрации `src/run_experiment.py` по-прежнему ожидает отдельный тяжёлый кэш.
