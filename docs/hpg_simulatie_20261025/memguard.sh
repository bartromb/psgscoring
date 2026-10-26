#!/bin/bash
# memguard.sh <unit> <log> [min_gb=12] — stopt de unit als MemAvailable onder min_gb zakt.
# Reden: de U-Net-schaalketen (25-10) houdt ~90 GB resident naast deze run; de Z6 bevroor
# eerder onder last zonder kernellogregel. Elke meting wordt gesynct weggeschreven.
set -u
UNIT=$1; LOG=$2; MIN=${3:-12}
while systemctl --user is-active --quiet "$UNIT"; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  echo "$(date '+%F %T') mem_avail=${a}G" >> "$LOG"; sync -d "$LOG" 2>/dev/null
  if [ -n "$a" ] && [ "$a" -lt "$MIN" ]; then
    echo "$(date '+%F %T') STOP: mem_avail ${a}G < ${MIN}G" >> "$LOG"; sync -d "$LOG"
    systemctl --user stop "$UNIT"; break
  fi
  sleep 10
done
echo "$(date '+%F %T') einde (unit niet meer actief)" >> "$LOG"
