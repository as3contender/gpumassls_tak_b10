#!/usr/bin/env bash
set -euo pipefail

# ==== локальные настройки ====
# Подхватим переменные из .env (если есть) в текущей директории (repo/)
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi

USER_ON_HOST="${DEPLOY_USER:-${USER_ON_HOST:-ubuntu}}"   # пример: DEPLOY_USER=denis ./deploy_gpu.sh
HOST="${DEPLOY_HOST:-${HOST:-127.0.0.1}}"
SSH="$USER_ON_HOST@$HOST"

REPO_URL="https://github.com/as3contender/gpumassls_tak_b10.git"
PORT="${DEPLOY_PORT:-8000}"
PROFILE="gpu"

# локальная папка с моделями (запускать из repo/)
MODELS_SRC="./models"

# ---- rsync прогресс флаг ----
if rsync --version 2>/dev/null | head -1 | grep -E 'version (3|4)\.' >/dev/null; then
  RSYNC_PROGRESS_FLAG="--info=progress2"
else
  RSYNC_PROGRESS_FLAG="--progress"
  echo "[warn] У тебя старая версия rsync. Рекомендуется: brew install rsync"
fi
command -v rsync >/dev/null || { echo "Установи rsync (brew install rsync)"; exit 1; }

echo "==> Подготовка сервера $SSH (Docker, compose, UFW, NVIDIA)"
ssh -o StrictHostKeyChecking=accept-new "$SSH" bash -s <<'REMOTE'
set -e
APP_DIR="$HOME/app"
REPO_DIR="$APP_DIR/repo"
PORT="${PORT:-8000}"

sudo apt-get update -y

# Docker + compose plugin
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
fi
sudo apt-get install -y docker-compose-plugin git curl ca-certificates ufw jq

# NVIDIA драйверы + Container Toolkit (если нужно)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] Устанавливаю NVIDIA драйверы..."
  # Под Ubuntu 22.04/24.04 драйвер 535 обычно ок; при желании поменяй на 550/555
  sudo apt-get install -y nvidia-driver-535 nvidia-utils-535 || true
  echo "[info] Устанавливаю NVIDIA Container Toolkit..."
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  echo "[note] Если nvidia-smi всё ещё не виден — возможно, нужна перезагрузка: sudo reboot"
fi

sudo systemctl enable --now docker

# каталоги
mkdir -p "$APP_DIR"
sudo ufw allow "${PORT}/tcp" || true

echo "REMOTE HOME=$HOME"
echo "REMOTE APP_DIR=$APP_DIR"
echo "REMOTE REPO_DIR=$REPO_DIR"
echo "NVIDIA GPU info (host):"
nvidia-smi || echo "nvidia-smi ещё недоступен (нужен драйвер или reboot)"
REMOTE

echo "==> Код: clone/pull $REPO_URL"
ssh "$SSH" bash -s <<'REMOTE'
set -e
APP_DIR="$HOME/app"
REPO_DIR="$APP_DIR/repo"
if [ -d "$REPO_DIR/.git" ]; then
  cd "$REPO_DIR"
  git fetch --all
  git reset --hard origin/main || git reset --hard origin/master || true
else
  mkdir -p "$APP_DIR"
  cd "$APP_DIR"
  git clone "https://github.com/as3contender/gpumassls_tak_b10.git" repo
fi
REMOTE

echo "==> Создание директории под модели на сервере"
ssh "$SSH" bash -s <<'REMOTE'
set -e
APP_DIR="$HOME/app"
mkdir -p "$APP_DIR/models"
REMOTE

echo "==> Передача моделей rsync: $MODELS_SRC -> $SSH:~/app/models/"
rsync -avhP --delete $RSYNC_PROGRESS_FLAG "$MODELS_SRC"/ "$SSH:~/app/models/"

echo "==> Проверка передачи моделей на сервер"
ssh "$SSH" bash -s <<'REMOTE'
set -e
APP_DIR="$HOME/app"
echo "Проверяем наличие моделей в $APP_DIR/models/ner_xlmr_wordlevel_entity/"
ls -la "$APP_DIR/models/ner_xlmr_wordlevel_entity/" | head -10 || echo "Модели не найдены!"
echo "Проверяем labels.txt:"
ls -la "$APP_DIR/models/ner_xlmr_wordlevel_entity/labels.txt" || echo "labels.txt не найден!"
REMOTE

echo "==> Проверка доступа к GPU из Docker"
ssh "$SSH" bash -s <<'REMOTE'
set -e
docker run --rm --gpus all nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 nvidia-smi || {
  echo "[error] Контейнер не видит GPU. Проверь драйвер/ctk, возможно нужен reboot."
  exit 1
}
REMOTE

echo "==> Сборка и запуск docker compose (профиль: $PROFILE)"
ssh "$SSH" bash -s <<'REMOTE'
set -e
APP_DIR="$HOME/app"
REPO_DIR="$APP_DIR/repo"
cd "$REPO_DIR"

# Приводим volume с моделями к абсолютному пути на сервере
MODELS_HOST_DIR="$HOME/app/models"
if grep -q ':/models' docker-compose.yml; then
  sed -i -E "s#-\s*[^[:space:]]*/models:/models(:rw)?#- ${MODELS_HOST_DIR}:/models:rw#g" docker-compose.yml || true
fi

# Чистим dangling, но не удаляем все образы
docker system prune -f || true

docker compose --profile gpu down || true
docker compose --profile gpu build --no-cache --pull
docker compose --profile gpu up -d
docker compose --profile gpu ps

echo "Проверяем volume внутри контейнера:"
docker compose --profile gpu exec ner-gpu ls -la /models/ner_xlmr_wordlevel_entity/ | head -10 || echo "Volume не подключен!"
echo "Проверяем labels.txt в контейнере:"
docker compose --profile gpu exec ner-gpu ls -la /models/ner_xlmr_wordlevel_entity/labels.txt || echo "labels.txt не найден в контейнере!"
REMOTE

echo "==> Smoke-тест с сервера"
ssh "$SSH" bash -s <<'REMOTE'
set -e
APP_DIR="$HOME/app"
REPO_DIR="$APP_DIR/repo"
cd "$REPO_DIR"
sleep 5
echo 'waiting for health...'
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health >/dev/null; then
    break
  fi
  sleep 2
  if [ "$i" -eq 60 ]; then
    docker compose --profile gpu logs --tail=200 || true
    exit 1
  fi
done
echo
echo 'health: OK'
echo '--- sample predict ---'
TMP_RESP="/tmp/predict_resp.json"
TMP_HDRS="/tmp/predict_hdrs.txt"
set +e
curl -sS -D "$TMP_HDRS" -o "$TMP_RESP" -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"input":"кефир 3.2% простоквашино 930 мл"}'
CURL_CODE=$?
set -e
if [ $CURL_CODE -ne 0 ]; then
  echo "curl failed with code $CURL_CODE"
  echo "--- headers ---"; cat "$TMP_HDRS" || true
  echo "--- body ---"; cat "$TMP_RESP" || true
  docker compose --profile cpu logs --tail=200 || true
  exit 1
fi
if jq -e . "$TMP_RESP" >/dev/null 2>&1; then
  jq . "$TMP_RESP"
else
  echo "[warn] predict ответ не JSON. Печатаю сырой вывод:"
  echo "--- headers ---"; cat "$TMP_HDRS" || true
  echo "--- body ---"; cat "$TMP_RESP" || true
  # не считаем это фатальным, но подсветим логи
    docker compose --profile gpu logs --tail=100 || true
fi
echo
REMOTE

echo "✅ Деплой завершён. API: http://$HOST:$PORT"