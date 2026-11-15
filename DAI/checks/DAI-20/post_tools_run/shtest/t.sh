#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'touch failed' ERR
rm -f done failed
snap() { { find nonexist_dir -type f -print0 | sort -z | xargs -0 sha256sum; find . -maxdepth 1 -name 't.sh' -printf '%p %s\n' | sort; } | sha256sum | cut -d' ' -f1; }
B=$(snap); echo "snap は通った：$B"
/home/snics/MER-DA/.venv/bin/python -c "import sys; print('py'); sys.exit('わざと失敗')" 2>&1 | tee -a log.txt
echo "ここに来てはいけない"
date > done
