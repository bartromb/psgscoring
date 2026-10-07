#!/bin/bash
# Start de MESA-run van docs/breath_dual_mesa_preregistratie_20261007.md in een eigen sessie,
# bind de temperatuurbewaker aan de ECHTE pgid en controleer dat die logt.
set -u
cd /srv/CODE/psgscoring
OUT=docs/breath_dual_mesa_20261007
W=${1:-20}
setsid nohup .venv/bin/python scripts/validate_mesa.py \
  --data-dir /srv/DATA/MESA/mesa \
  --recordings $(cat $OUT/opnames.txt | tr '\n' ' ') \
  --profiles aasm_v3_breath aasm_v3_breath_dual --strictness 0.5 0.3 --dual-confirmation \
  --workers "$W" --output-json $OUT/mesa.json > $OUT/run.log 2>&1 < /dev/null &
sleep 5
PID=$(pgrep -f "validate_mesa.py --data-dir /srv/DATA/MESA/mesa --recordings" | head -1)
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
echo "validate_mesa pid=$PID pgid=$PGID"
nohup /srv/CODE/docs/arousal_unet_20260927/thermal_pauze.sh "$PGID" $OUT/thermal.log 81 68 10 > /dev/null 2>&1 < /dev/null &
sleep 25
echo "bewaker:"; tail -2 $OUT/thermal.log
echo "run.log:"; head -5 $OUT/run.log
