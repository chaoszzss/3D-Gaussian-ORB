#!/usr/bin/env bash
# Run only after the user explicitly resumes the paused reconstruction.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/python3 scripts/reconstruct_garage.py \
  --run-id garage_surface_v1 --renderer-inria --surface-splats \
  --iterations 12000 --depth-weight 0.5 --alpha-weight 0.1 \
  --prepare-workers 2 >> runtime/garage_inria_training.log 2>&1
