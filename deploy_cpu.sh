#!/usr/bin/env bash
set -euo pipefail

# ==== настройка ====
USER="gpumasslsroot"          # <= ИЗМЕНИ при необходимости: имя пользователя на сервере
HOST="89.169.187.74"            # адрес сервера из твоего сообщения
SSH="$USER@$HOST"

REPO_URL="https://github.com/as3contender/gpumassls_tak_b10.git"
APP_DIR="$HOME/app"             # корневая папка на сервере: ~/app
REPO_DIR="$APP_DIR/repo"
MODELS_SRC="../models"          # локальная папка с моделями (относительно текущего каталога)
PORT="8000"                     # наружный порт API
PROFILE="cpu"                   # профиль docker-compose

# ==== проверки ====
if ! command -v rsync >/dev/null 2>&1; then
  echo "Установи rsync (macOS: brew install rsync)"; exit 1
fi

echo "==> Подготовка сервера $SSH"
ssh -o StrictHostKeyChecking=accept-new "$SSH" bash -lc "
  set -e
  sudo apt-get update -y
  # Docker
  if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker \$USER || true
  fi
  # docker compose plugin
  sudo apt-get install -y docker-compose-plugin
  # базовые утилиты
  sudo apt-get install -y git curl ca-certificates ufw
  # директории
  mkdir -p $APP_DIR
  # открыть порт
  sudo ufw allow ${PORT}/tcp || true
  sudo systemctl enable --now docker
"

echo "==> Код: clone/pull $REPO_URL -> $REPO_DIR"
ssh "$SSH" bash -lc "
  set -e
  if [ -d '$REPO_DIR/.git' ]; then
    cd '$REPO_DIR' && git fetch --all && git reset --hard origin/main || true
  else
    mkdir -p '$APP_DIR'
    cd '$APP_DIR'
    git clone '$REPO_URL' repo
  fi
"

echo '==> Ссылка/монтирование models'
# На нашем compose локально папка models находится рядом с repo (../models).
# Разложим на сервере так же: ~/app/{repo,models}
ssh "$SSH" bash -lc "mkdir -p '$APP_DIR/models'"

echo "==> Передача моделей rsync: $MODELS_SRC -> $SSH:$APP_DIR/models/"
rsync -avhP --delete --info=progress2 "$MODELS_SRC"/ "$SSH":"$APP_DIR/models"/

echo "==> Сборка и запуск docker compose (профиль: $PROFILE)"
ssh "$SSH" bash -lc "
  set -e
  cd '$REPO_DIR'
  # убедимся, что compose смотрит в ../models
  if ! grep -q '../models:/models' docker-compose.yml && grep -q './models:/models' docker-compose.yml; then
    # если у тебя в compose ещё ./models — попробуем переключить на ../models
    sed -i 's#\./models:/models#../models:/models#g' docker-compose.yml || true
  fi

  docker compose --profile $PROFILE down || true
  docker compose --profile $PROFILE build --no-cache
  docker compose --profile $PROFILE up -d
  docker compose --profile $PROFILE ps
"

echo "==> Smoke-тест с сервера"
ssh "$SSH" bash -lc "
  set -e
  sleep 2
  curl -sf http://localhost:${PORT}/healthz || (docker compose --profile $PROFILE logs --tail=200 && exit 1)
  echo
  echo 'health: OK'
  echo '--- sample predict ---'
  curl -s -X POST http://localhost:${PORT}/predict \
    -H 'Content-Type: application/json' \
    -d '{\"texts\":[\"кефир 3.2% простоквашино 930 мл\"]}' | jq .
  echo
  echo 'Готово: http://$HOST:${PORT}'
"

echo "✅ Деплой завершён. API: http://$HOST:$PORT"