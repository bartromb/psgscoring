#!/bin/bash
# start.sh — hPG-simulatie op MESA (150 nachten, 3 armen) onder het recept van 16-08-2026:
# systemd --user unit met CPUQuota=300 %, 3 workers (naast de U-Net-keten van ~90 GB RSS), één thread per proces, en de thermische
# bewaker (scripts/thermal_guard.sh) als tweede unit die de run stopt bij 3× ≥ 78 °C.
# Preregistratie: docs/hpg_simulatie_preregistratie_20261025.md (sha256 in run.log).
set -eu
REPO=/srv/CODE/psgscoring
OUT=$REPO/docs/hpg_simulatie_20261025
UNIT=hpg-simulatie-20261025
PY=$REPO/.venv/bin/python
{
  echo "start $(date '+%F %T')"
  echo "prereg sha256 $(sha256sum $REPO/docs/hpg_simulatie_preregistratie_20261025.md | cut -d' ' -f1)"
  echo "git HEAD $(git -C $REPO rev-parse --short HEAD) (werkboom met ongecommitte wijzigingen van 25-10: unet_v1-default, agreement, hr_surge)"
  echo "psgscoring $($PY -c 'import psgscoring; print(psgscoring.__version__)')"
} > $OUT/run.log
systemd-run --user --unit="$UNIT" -p CPUQuota=300% \
  --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=NUMEXPR_NUM_THREADS=1 --setenv=PSGSCORING_UNET_THREADS=1 \
  --working-directory="$REPO" \
  /bin/bash -c "$PY scripts/hpg_simulatie_mesa.py --ids-file $OUT/ids.txt --out $OUT --workers 3 >> $OUT/run.log 2>&1; echo \"KLAAR \$? \$(date '+%F %T')\" > $OUT/KLAAR"
LOG=$OUT/thermal_guard.log systemd-run --user --unit="${UNIT}-guard" \
  --setenv=LOG=$OUT/thermal_guard.log "$REPO/scripts/thermal_guard.sh" "$UNIT" 78 3 10
# Geheugenbewaker: de U-Net-schaalketen houdt ~90 GB resident; onder 12 GB beschikbaar stopt
# deze unit de hPG-run (de kleinste van de twee) vóór de kernel iets killt of de machine bevriest.
systemd-run --user --unit="${UNIT}-memguard" "$OUT/memguard.sh" "$UNIT" "$OUT/memguard.log" 12
sleep 5
systemctl --user --no-pager status "$UNIT" "${UNIT}-guard" "${UNIT}-memguard" | grep -E 'Active|unit' || true
