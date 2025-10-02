# 🚀 NER Web Service

## 📌 Описание
Сервис для **распознавания сущностей в тексте** (бренды, типы товаров, характеристики). Возвращает список интервалов сущностей с типами.

- Ввод через REST API
- Высокопроизводительный батчинг (очередь) для нагрузки
- Постпроцессинг правил для повышения качества

---

## 🛠 Технологии
- Python 3.10+
- FastAPI + Uvicorn
- Transformers (токенизация) + ONNX Runtime (инференс)
- Docker, docker-compose (GPU)

Модель и токенайзер лежат в `./models/ner_xlmr_wordlevel_entity` (рядом с этим README).

---

## ⚙️ Требования
- Linux/Mac/Windows
- Python 3.10+
- Опционально: NVIDIA GPU + Docker/NVIDIA Container Toolkit

Минимальные ресурсы: 2 CPU, 4 GB RAM, ~2 GB диска.

---

## ⚡ Быстрый старт (Docker, CPU)
```bash
# Клонируем репозиторий и подтягиваем модель через Git LFS
git clone <URL_РЕПОЗИТОРИЯ>
git lfs install && git lfs pull

# Поднимаем сервис в Docker (CPU профиль)
docker compose --profile cpu up --build -d

# Проверка
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"input":"кефир 3.2% простоквашино 930 мл"}'
```

---

## 🚀 Быстрый старт (локально, CPU)
Запускайте из директории `repo/`.

```bash
cd repo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.cpu.txt

# Рекомендуется отключить параллелизм токенайзера HF
export TOKENIZERS_PARALLELISM=false

# Запуск API (объект приложения: app.main:api)
uvicorn app.main:api --host 0.0.0.0 --port 8000
```

Проверка здоровья:
```bash
curl http://localhost:8000/health
```

Запрос предсказаний:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"input":"Молоко Простоквашино 2% 1л"}'
```

Ответ (пример):
```json
[
  {"start_index": 0, "end_index": 6, "entity": "BRAND"}
]
```

Примечания:
- По умолчанию сервис возьмёт модель из `./models/ner_xlmr_wordlevel_entity`.
- Можно переопределить через переменные окружения `MODEL_DIR`, `LABELS_PATH`.

---

## 🧰 Пакетная инференция (CLI)
Есть утилита для обработки CSV пачками (`repo/app/infer.py`).

Вариант A (запуск из корня репозитория):
```bash
python -m repo.app.infer \
  --csv-in train/data/submission.csv \
  --csv-out submission_pred.csv \
  --batch-size 16
```

Вариант B (запуск из `repo/`):
```bash
cd repo
python -m app.infer --csv-in ../train/data/submission.csv --csv-out ../submission_pred.csv --batch-size 16
```

Выходной CSV содержит колонки: `sample;annotation`.

---

## 🔎 Troubleshooting
- Модель не подтянулась (пустой `model.onnx`): убедитесь, что установлен Git LFS и выполнено `git lfs pull` в `repo/`.
- Порт 8000 занят: поменяйте порт в команде запуска или остановите конфликтующий процесс.
- Медленный старт: первый запуск прогревает токенизатор и граф ONNX; дальнейшие запросы быстрее.

## 🐳 Docker (CPU)
Рекомендуется собирать из директории `repo/` с указанием Dockerfile внутри `repo/docker`.

```bash
cd repo
docker build -t ner-cpu -f docker/Dockerfile.cpu .

# Монтируем модель внутрь контейнера в /models
docker run --rm -p 8000:8000 \
  -e TOKENIZERS_PARALLELISM=false \
  -e MODEL_DIR=/models/ner_xlmr_wordlevel_entity \
  -e LABELS_PATH=/models/ner_xlmr_wordlevel_entity/labels.txt \
  -v "$(pwd)/models:/models:ro" \
  ner-cpu
```

---

## ⚡️ Docker (GPU)
Требуется NVIDIA драйвер и NVIDIA Container Toolkit.

```bash
cd repo
docker build -t ner-gpu -f docker/Dockerfile.gpu .

docker run --rm -p 8000:8000 --gpus all \
  -e TOKENIZERS_PARALLELISM=false \
  -e MODEL_DIR=/models/ner_xlmr_wordlevel_entity \
  -e LABELS_PATH=/models/ner_xlmr_wordlevel_entity/labels.txt \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$(pwd)/models:/models:ro" \
  ner-gpu
```

Альтернатива — docker-compose (GPU профиль):
```bash
cd repo
docker compose --profile gpu up --build -d
```

Compose-файл автоматически смонтирует `./models` в `/models` и выставит нужные переменные.

---

## 🧩 docker-compose (CPU профиль)
```bash
cd repo
docker compose --profile cpu up --build -d
```

Сервис использует `docker/Dockerfile.cpu`, монтирует локальные `./models` в контейнер `/models` и пробрасывает порт `8000`.

---

## 🔧 Полезные переменные окружения
- `MODEL_DIR` — путь к директории модели (по умолчанию: локально `./models/ner_xlmr_wordlevel_entity`, в контейнере ожидается `/models/...`).
- `LABELS_PATH` — путь к `labels.txt`.
- `MAX_SEQ_LEN` — максимальная длина последовательности (по умолчанию 256 локально; в compose пример 64).
- `USE_QUEUE` — включить батчинг запросов (`true/false`).
- `BATCH_MAX_SIZE`, `BATCH_TIMEOUT_MS` — параметры батчинга.
- `REQUEST_TIMEOUT_MS` — таймаут обработки запроса.
- `ORT_FORCE_CPU` — форсировать CPU в ONNX Runtime (`1`/`0`).
- `TOKENIZERS_PARALLELISM` — выключить варнинги HF токенайзера (`false`).

---

## 🚚 Деплой-скрипты и .env
Скрипты `deploy_cpu.sh` и `deploy_gpu.sh` читают конфиг из файла `.env` в папке `repo/` (если он существует):

Переменные:
- `DEPLOY_USER` — пользователь на сервере (например, `ubuntu`)
- `DEPLOY_HOST` — адрес сервера (например, `1.2.3.4`)
- `DEPLOY_PORT` — порт публикации API (по умолчанию `8000`)

Пример `.env`:
```
DEPLOY_USER=ubuntu
DEPLOY_HOST=1.2.3.4
DEPLOY_PORT=8000
```

Запуск (из `repo/`):
```bash
./deploy_cpu.sh   # или ./deploy_gpu.sh
```


## 🧪 Эндпоинты
- `GET /health` — статус сервиса
- `POST /api/predict` — тело: `{ "input": "строка" }`; ответ — список `{ start_index, end_index, entity }`.

---

## ✅ Тестирование
Быстрые unit-тесты (модель не загружается, подменяется стабом):
```bash
cd repo
python -m pip install pytest
pytest -q
```

---

## 📚 EDA и обучение
- EDA и подготовка данных: см. ноутбук `repo/train/EDA_GENERAYION.ipynb` (анализ распределений, длины текстов, частоты меток, sanity-check разметки).

- Обучение модели: см. ноутбук `repo/train/final-v2 (1).ipynb`.
  - Базовая модель: `xlm-roberta-large` (Token Classification)
  - Токенайзер: `AutoTokenizer.from_pretrained(MODEL_NAME)`
  - Токенизация при обучении: `max_length=128`
  - Гиперпараметры (TrainingArguments):
    - `learning_rate=3e-5`
    - `per_device_train_batch_size=16`
    - `num_train_epochs=12`
  - Кастомный тренер: `FocalTrainerWithOffsets(Trainer)` + `DataCollatorForTokenClassification`
  - Экспорт артефактов:
    - чекпоинт и токенайзер → `./ner_xlmr_wordlevel_entity_best`
    - TorchScript → `model_torchscript.pt`
    - ONNX → `model.onnx`
- Инференс:
  - В ноутбуке: `MAX_LENGTH=512`; в сервисе: `max_seq_len=256` (см. `app/settings.py`) для скорости.
  - В проде используется `ONNX Runtime` (см. `app/runtime.py`), модель из `repo/models/ner_xlmr_wordlevel_entity`.

## 🏗 Архитектура (вкратце)
1. Текст → нормализация/маскирование (по настройкам)
2. Токенизация (Transformers)
3. Инференс ONNX Runtime (CPU/GPU)
4. Постпроцессинг правил (опционально)
5. JSON-ответ со списком сущностей


