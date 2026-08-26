#!/bin/bash
# Gepaarde A/B van `arousal_onset_offset_s`: dezelfde opnames, dezelfde
# pijplijn, alleen de verschuiving verschilt. Arm 0 s eerst, dan arm +2 s.
#
# De verschuiving komt uit sitecustomize.py hiernaast en NIET uit de
# profielvlag, zodat een oudere psgscoring op dezelfde manier gemeten kan
# worden. Python importeert sitecustomize in ELK proces, dus ook in de workers
# van validate_mesa -- een patch in het ouderproces alleen is bij `spawn`
# stilzwijgend weg en levert twee identieke armen op.
#
# Workers: elke worker is op één draad gepind. Op de Z6 gaf 6 workers 66-69 C,
# 8 gaf 70-72 C en 12 gaf 79 C -- die laatste is te heet (crit 84, en de
# machine bevroor eerder rond 82). validate_mesa hervat uit zijn .partial.jsonl,
# dus onderbreken en met minder workers herstarten kost niets.
#
#   ./run_ab.sh [uitvoermap]     (default ./mesa_onset_ab)
set -u
HIER="$(cd "$(dirname "$0")" && pwd)"
UIT="${1:-$PWD/mesa_onset_ab}"
mkdir -p "$UIT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$HIER":"$(cd "$HIER/../.." && pwd)"
REPO="$(cd "$HIER/../.." && pwd)"
PY="${PSGSCORING_PYTHON:-$REPO/.venv/bin/python}"
cd "$REPO" || exit 1
for OFF in 0 2; do
  echo "=================== offset ${OFF}s  ($(date +%H:%M:%S)) ==================="
  PSGSCORING_AROUSAL_ONSET_OFFSET_S=$OFF \
  PSGSCORING_AROUSAL_ONSET_MARKFILE=$UIT/mark_off${OFF}.txt \
  $PY -u scripts/validate_mesa.py \
      --data-dir "${MESA_DIR:-$HOME/MESA/mesa}" \
      --n 30 --workers 8 --seed 20260801 \
      --profiles aasm_v3_breath \
      --output-json $UIT/mesa_off${OFF}.json 2>&1
  RC=$?
  echo "--- offset ${OFF}s exit=${RC} $(date +%H:%M:%S); markeringen: $(wc -l < $UIT/mark_off${OFF}.txt 2>/dev/null || echo 0)"
  if [ ! -s "$UIT/mesa_off${OFF}.json" ]; then
    echo "!! offset ${OFF}s heeft GEEN output-json geschreven - gestopt, arm 2 niet gestart"
    exit 1
  fi
done
echo "AB KLAAR $(date +%H:%M:%S)"
