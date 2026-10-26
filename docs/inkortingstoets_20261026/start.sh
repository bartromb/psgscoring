#!/bin/bash
# start.sh — inkortingstoets op MESA (30 nachten, 6 armen), recept 25-10: systemd --user unit met CPUQuota 800 %,
# 8 workers, één thread per proces, thermische bewaker (scripts/thermal_guard.sh) 78 °C / 3 / 10 s.
set -eu
REPO=/srv/CODE/psgscoring; OUT=$REPO/docs/inkortingstoets_20261026; UNIT=inkortingstoets-20261026; PY=$REPO/.venv/bin/python
{ echo "start $(date '+%F %T')"; echo "prereg sha256 $(sha256sum $REPO/docs/inkortingstoets_preregistratie_20261026.md | cut -d' ' -f1)"; echo "git HEAD $(git -C $REPO rev-parse --short HEAD)"; echo "psgscoring $($PY -c 'import psgscoring; print(psgscoring.__version__)')"; } > $OUT/run.log
systemd-run --user --unit="$UNIT" -p CPUQuota=800% \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  --setenv=PSGSCORING_AROUSAL_DETECTOR=lgbm --working-directory="$REPO" \
  /bin/bash -c "$PY scripts/inkortingstoets_mesa.py --ids-file $OUT/ids.txt --out $OUT --workers 8 >> $OUT/run.log 2>&1; echo \"KLAAR \$? \$(date '+%F %T')\" > $OUT/KLAAR"
LOG=$OUT/thermal_guard.log systemd-run --user --unit="${UNIT}-guard" --setenv=LOG=$OUT/thermal_guard.log "$REPO/scripts/thermal_guard.sh" "$UNIT" 78 3 10
sleep 5; systemctl --user --no-pager status "$UNIT" "${UNIT}-guard" | grep -E 'Active|unit' || true
