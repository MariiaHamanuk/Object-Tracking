#!/usr/bin/env bash
# Підключає форк OSTrack як git submodule у baselines/OSTrack.
# Спочатку зробіть форк https://github.com/botaoye/OSTrack на GitHub.
# Використання: bash scripts/add_ostrack_submodule.sh [URL форку]
set -euo pipefail

FORK_URL="${1:-https://github.com/MariiaHamaniuk/OSTrack.git}"

cd "$(dirname "$0")/.."
rm -f baselines/.gitkeep
git submodule add "$FORK_URL" baselines/OSTrack
git -C baselines/OSTrack remote add upstream https://github.com/botaoye/OSTrack.git
echo "Готово. Не забудьте закомітити .gitmodules і baselines/OSTrack."
