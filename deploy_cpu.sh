#!/usr/bin/env bash
set -euo pipefail

# ==== настройка (локальная) ====
USER_ON_HOST="${USER_ON_HOST:-ubuntu}"             # свой юзер на сервере, можно USER_ON_HOST=denis ./deploy_cpu.sh
HOST="89.169.187.74"
SSH="$USER_ON_HOST@$HOST"

REPO_URL="https://github.com/as3contender/gpumassls_tak_b10.git"
PORT="8000"
PROFILE="cpu"

# локальная папка с моделями: структура как у тебя — на уровень выше repo/
MODELS_SRC="../models"   # запускай скрипт из repo/, чтобы путь был валиден

# ==== проверки локально ====
command -v rsync >/dev/null || { echo "Установи rsync (brew install rsync)"; exit 1; }

# Определим совместимый флаг прогресса для rsync
# У старого rsync из macOS нет --info=progress2, поэтому подменяем на --progress
if rsync --version 2>/dev/null | head -1 | grep -E 'version (3|4)\.' >/dev/null; then
  RSYNC_PROGRESS_FLAG="--info=progress2"
else
  RSYNC_PROGRESS_FLAG="--progress"
  echo "[warn] У тебя старая версия rsync. Рекомендуется: brew install rsync"
fi

echo "==> Подготовка сервера $SSH (Docker, compose, ufw, каталоги)"
ssh -o StrictHostKeyChecking=accept-new "$SSH" bash -s <<'REMOTE'
set -e
# Переменные только на УДАЛЕННОЙ стороне
APP_DIR="$HOME/app"
REPO_DIR="$APP_DIR/repo"
PORT="8000"

sudo apt-get update -y
# Docker + compose plugin (если ещё нет)
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
fi
sudo apt-get install -y docker-compose-plugin git curl ca-certificates ufw rsync jq
sudo systemctl enable --now docker

# каталоги приложения
mkdir -p "$APP_DIR"
# открыть порт (если ufw включён — правило добавится; если выключен — не помешает)
sudo ufw allow "${PORT}/tcp" || true

# Выведем для отладки
echo "REMOTE HOME=$HOME"
echo "REMOTE APP_DIR=$APP_DIR"
echo "REMOTE REPO_DIR=$REPO_DIR"
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
# (эта команда исполняется ЛОКАЛЬНО, тут как раз нужен локальный $MODELS_SRC)
rsync -avhP --delete $RSYNC_PROGRESS_FLAG "$MODELS_SRC"/ $SSH:~/app/models/

echo "==> Сборка и запуск docker compose (профиль: $PROFILE)"
ssh "$SSH" bash -s <<'REMOTE'
set -e
APP_DIR="$HOME/app"
REPO_DIR="$APP_DIR/repo"
cd "$REPO_DIR"

# Нормализуем volume: абсолютный путь до моделей на хосте
MODELS_HOST_DIR="$HOME/app/models"
# Заменим любой левый путь вида "- something/models:/models[:rw]" на абсолютный
if grep -q ':/models' docker-compose.yml; then
  sed -i -E "s#-\s*[^[:space:]]*/models:/models(:rw)?#- ${MODELS_HOST_DIR}:/models:rw#g" docker-compose.yml || true
fi

docker compose --profile cpu down || true
# Полная очистка образов и кеша
docker system prune -f || true
docker image prune -a -f || true
# Пересборка без кеша
docker compose --profile cpu build --no-cache --pull
docker compose --profile cpu up -d
docker compose --profile cpu ps
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
  if curl -sf http://localhost:8000/healthz >/dev/null; then
    break
  fi
  sleep 2
  if [ "$i" -eq 60 ]; then
    docker compose --profile cpu logs --tail=200 || true
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
  docker compose --profile cpu logs --tail=100 || true
fi
echo
REMOTE

echo "✅ Деплой завершён. API: http://$HOST:$PORT"