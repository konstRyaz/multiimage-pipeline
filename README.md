# Конвейер лиц из видеокадров

Актуальное состояние проекта, результаты запусков и список незавершённых задач
находятся в [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md). Порядок развёртывания на
новой машине описан в [`RECOVERY.md`](RECOVERY.md).

Проект восстанавливает `process_faces.py` и реализует все следующие этапы:

1. детекция, выравнивание и 512-мерные эмбеддинги InsightFace `buffalo_l`;
2. диагностика сходства и качества;
3. временные дорожки лиц внутри каждого видео;
4. кластеризация дорожек в личности;
5. обзорные листы для проверки;
6. ручные поправки на уровне дорожек;
7. удаление почти одинаковых кадров и выбор качественных разнообразных лиц;
8. сохранение крупных кропов из исходных кадров только для выбранных лиц.

Для воспроизводимых экспериментов фильтрации добавлен отдельный совместимый
контур. Он не меняет тяжёлые результаты и позволяет сравнивать политики без
повторного запуска InsightFace:

```text
faces.csv + embeddings.npy + aligned_faces/
  → независимые признаки
  → политика off/shadow/hard
  → жёсткая фильтрация
  → дорожки и кластеризация
  → мягкий рейтинг
  → дедупликация и разнообразие
```

Полное описание схем, конфигураций, результатов и команд находится в
[`docs/quality_experiments.md`](docs/quality_experiments.md).

Исходные `faces.csv`, `embeddings.npy` и `aligned_faces/` последующие этапы не
изменяют.

## Как подключить уже обработанные 663 лица

На сервере распакуйте проект и перейдите в его каталог. Если каталог
`/home/k.ryazanov/multiimage_pipeline` вместе с `runs/` сохранился, достаточно
перенести в него папки `face_pipeline`, `src` и файлы зависимостей из
архива. Каталог `runs/what_people_enjoy_full` копировать или удалять не нужно.

В существующем окружении установите только лёгкие зависимости следующих
этапов:

```bash
cd /home/k.ryazanov/multiimage_pipeline
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Строгая проверка входа без записи результатов:

```bash
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, ".")
from face_pipeline.io import load_face_run

rows, embeddings = load_face_run(Path("runs/what_people_enjoy_full"), require_images=True)
print("Строк:", len(rows))
print("Эмбеддинги:", embeddings.shape)
print("Входные данные согласованы")
PY
```

Полный запуск после `process_faces.py`:

```bash
python src/run_pipeline.py \
  --run-dir runs/what_people_enjoy_full
```

Прежние команды остаются совместимыми и без конфигурации воспроизводят старое
поведение. Новый эксперимент запускается отдельно:

```bash
python src/run_experiment.py \
  --run-dir runs/what_people_enjoy_full \
  --config configs/baseline_v1.json
```

Сравнение двух целевых конфигураций на одном кэше:

```bash
python src/compare_experiments.py \
  --run-dir runs/what_people_enjoy_full \
  --configs \
    configs/baseline_v1.json \
    configs/hard_filter_v1_soft_ranking_v1.json
```

Стартовые значения `0.38` для дорожек и `0.45` для личностей намеренно вынесены
в аргументы. Это не универсальные пороги. Сначала посмотрите
`analysis/summary.json`, `analysis/same_frame_pairs.csv` и обзорные листы
кластеров. Для повторного прогона с другими значениями:

```bash
python src/run_pipeline.py \
  --run-dir runs/what_people_enjoy_full \
  --track-threshold 0.40 \
  --cluster-threshold 0.50 \
  --overwrite
```

Флаг `--overwrite` удаляет и заново создаёт только производные каталоги
`analysis/`, `tracking/`, `clustering/` и `selected/`. Исходные результаты
`process_faces.py` не затрагиваются.

## Восстановленный `process_faces.py`

На машине, где уже установлены InsightFace, ONNX Runtime и модель `buffalo_l`,
прежняя команда сохраняется:

```bash
CUDA_VISIBLE_DEVICES=0 python src/process_faces.py \
  --frames-dir "/home/t.chichkanov/ffmpeg-7.0.2-amd64-static/output/frames" \
  --pattern "05 2 What Do People Really Enjoymp4_frame_*.jpg" \
  --output-dir "runs/what_people_enjoy_full"
```

Для нового окружения зависимости входного этапа находятся в
`requirements-insightface.txt`. Первый запуск InsightFace может скачать модель,
если её ещё нет в `~/.insightface/models/buffalo_l`.

Сценарий сохраняет прежние основные поля и добавляет размеры исходного кадра и
отступ рамки от границы. Результат:

```text
RUN_DIR/
├── faces.csv
├── embeddings.npy
├── aligned_faces/
├── frame_stats.csv
└── run_config.json
```

Каждая строка `faces.csv` соответствует ровно одной строке нормализованной
матрицы `embeddings.npy`; связь задаёт `embedding_index`.

### Независимые фотографии из вложенных каталогов

Для обычных фотографий, а не последовательных видеокадров, обязательно
укажите `--input-type photos`:

```bash
python src/process_faces.py \
  --frames-dir /home/t.chichkanov/vk_fetcher/vk_photos \
  --pattern "**/*.jpg" \
  --input-type photos \
  --output-dir runs/vk_cpu_100 \
  --limit 100 \
  --device -1 \
  --provider cpu
```

В этом режиме каждая фотография считается независимым источником. Уникальный
идентификатор строится из относительного пути с коротким хешем, поэтому файлы
`2926/photo_1.jpg` и `6210/photo_1.jpg` не перезаписывают результаты друг
друга. На следующем этапе каждое лицо сначала образует отдельную дорожку, а
кластеризация уже объединяет похожие лица с разных фотографий. Лица с одной
фотографии не могут быть ошибочно объединены в одну личность.

## Что делает кластеризация

Сначала лица соседних кадров сопоставляются венгерским алгоритмом по сходству
эмбеддингов и пересечению рамок. Получаются дорожки — более устойчивые средние
эмбеддинги одного непрерывного появления человека.

Затем дорожки объединяются полным связыванием: все пары дорожек внутри итоговой
личности должны пройти порог. Дорожки, интервалы которых пересекаются в одном
видео, объединять запрещено. Поэтому два разных человека из одного кадра не
могут случайно стать одной личностью даже при похожих эмбеддингах.

Маленькие кластеры получают имена `unknown_NNN` и по умолчанию не попадают в
финальный отбор. Для ручных поправок скопируйте `corrections.example.csv`,
укажите одинаковую `identity_label` у дорожек одного человека и запустите:

```bash
python src/cluster_faces.py \
  --run-dir runs/what_people_enjoy_full \
  --corrections corrections.csv \
  --overwrite

python src/select_faces.py \
  --run-dir runs/what_people_enjoy_full \
  --overwrite
```

Ручная поправка не позволит объединить пересекающиеся дорожки и завершится с
понятной ошибкой, если нарушен этот инвариант.

## Структура итогов

```text
RUN_DIR/
├── analysis/
│   ├── summary.json
│   ├── similarity_histogram.csv
│   ├── nearest_neighbors.csv
│   ├── same_frame_pairs.csv
│   ├── suspicious_faces.csv
│   ├── threshold_sweep.csv
│   └── temporal_contact_sheets/
├── tracking/
│   ├── faces_tracked.csv
│   ├── tracks.csv
│   ├── track_embeddings.npy
│   └── summary.json
├── clustering/
│   ├── faces_clustered.csv
│   ├── clusters.csv
│   ├── cluster_embeddings.npy
│   ├── contact_sheets/
│   └── summary.json
└── selected/
    ├── selection.csv
    ├── summary.json
    └── person_NNN/
        ├── aligned/
        ├── source/
        └── contact_sheet.jpg
```

`source/` содержит кропы исходного разрешения с запасом вокруг головы. Они
создаются только для выбранных изображений, поэтому место не расходуется на
крупные кропы всех детекций.

## Ограничения, которые нужно проверить на реальных данных

- Пороги сходства зависят от модели и ваших видео; значения в проекте —
  безопасная исходная точка, а не окончательная калибровка.
- Поза из `process_faces.py` заполняется, только если пакет модели отдаёт
  `pose`; иначе поля остаются пустыми и не мешают остальным этапам.
- Запрет пересекающихся дорожек предполагает, что один человек не показан дважды
  в одном кадре, например одновременно в основном видео и во вставке.
