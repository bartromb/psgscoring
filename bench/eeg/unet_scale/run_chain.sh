#!/bin/bash
# Start de schaalketen in de achtergrond, in een eigen sessie/procesgroep, met de thermische
# bewaker (pauze bij 3× ≥ 78 °C, hervat ≤ 66 °C). Logs en markers in $OUT.
#   bash bench/eeg/unet_scale/run_chain.sh [extra train_scale.py-opties]
set -u
OUT=/srv/CODE/docs/arousal_unet_schaal_20261025
HIER=$(dirname "$(readlink -f "$0")")
PY=/srv/CODE/psgscoring/bench/eeg/_venv/bin/python
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
setsid nohup "$PY" "$HIER/train_scale.py" --out "$OUT" --workers 8 "$@" >> "$OUT/chain.log" 2>&1 &
PID=$!
echo "$PID" > "$OUT/chain.pid"
sleep 2
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
echo "keten pid $PID pgid $PGID gestart $(date '+%F %T')" | tee -a "$OUT/chain.log"
nohup bash "$HIER/thermal_pauze.sh" "$PGID" "$OUT/thermal.log" 78 66 10 >/dev/null 2>&1 &
echo $! > "$OUT/thermal.pid"
echo "thermische bewaker pid $(cat "$OUT/thermal.pid") op pgid $PGID"
