#!/usr/bin/env bash
set -euo pipefail

app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$app_dir/.." && pwd)"
cd "$repo_dir"

exec uv run --no-sync python "$app_dir/bot.py" -t webrtc --host localhost --port 7860 "$@"
