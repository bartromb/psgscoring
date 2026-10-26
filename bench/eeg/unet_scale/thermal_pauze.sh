#!/bin/bash
# thermal_pauze.sh <pgid> <log> [pauze_c=81] [hervat_c=68] [interval=10]
# Pauzeert (SIGSTOP) de hele procesgroep als de package-temperatuur 3 metingen op rij >= pauze_c is,
# hervat (SIGCONT) zodra ze <= hervat_c is. Elke meting gaat naar de log (laatste regel = stand vóór een freeze).
set -u
PG=$1; LOG=$2; MAX=${3:-81}; HERVAT=${4:-68}; INT=${5:-10}
n=0; gepauzeerd=0
while kill -0 -- -"$PG" 2>/dev/null; do
  t=$(( $(cat /sys/class/thermal/thermal_zone1/temp) / 1000 ))
  g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
  la=$(cut -d' ' -f1 /proc/loadavg)
  echo "$(date +%H:%M:%S) pkg=${t}C gpu=${g:-?}C load=${la} pauze=${gepauzeerd}" >> "$LOG"
  if [ "$gepauzeerd" -eq 0 ]; then
    if [ "$t" -ge "$MAX" ]; then n=$((n+1)); else n=0; fi
    if [ "$n" -ge 3 ]; then kill -STOP -- -"$PG"; gepauzeerd=1; n=0; echo "$(date +%H:%M:%S) PAUZE bij ${t}C" >> "$LOG"; fi
  else
    if [ "$t" -le "$HERVAT" ]; then kill -CONT -- -"$PG"; gepauzeerd=0; echo "$(date +%H:%M:%S) HERVAT bij ${t}C" >> "$LOG"; fi
  fi
  sleep "$INT"
done
echo "$(date +%H:%M:%S) procesgroep $PG beëindigd" >> "$LOG"
