#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PHASE1=1
export RUN_PHASE2=0

exec "$SCRIPT_DIR/train_full.bash" "$@"
