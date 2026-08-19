#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# thermal_guard.sh — stop een lange run vóór de machine bevriest
# ═══════════════════════════════════════════════════════════════
#
# WAAROM DIT BESTAAT
#
# Op 16-08-2026 bevroor de Z6 twee keer tijdens `validate_mesa.py`, allebei
# 20-30 minuten na de start, allebei zonder één regel in het kernellog. De
# eerste diagnose (geheugen) was FOUT: bpytop liet op het moment van de tweede
# freeze 57,8 GiB in gebruik zien met 62,1 GiB vrij en ongebruikte swap.
#
# Wat er wél aan de hand was: `sensors` geeft voor deze CPU
#
#     Package id 0:  high = +74,0°C, crit = +84,0°C
#
# en op de foto van de freeze stonden meerdere cores op 83 °C. Een Xeon
# Platinum 8273CL is een 205 W-SKU; aanhoudende all-core FFT-belasting is de
# heetste last die er bestaat en de koeling verzadigt pas na tientallen
# minuten — vandaar dat korte runs altijd goed gingen.
#
# Bij een thermische hardlock is er geen tijd om het journaal te flushen. Daar
# komt de tweede taak van dit script vandaan: het schrijft elke ronde een
# regel wég naar schijf, zodat de LAATSTE regel vóór een eventuele freeze
# vertelt hoe heet en hoe belast de machine op dat moment was. Zonder zoiets
# blijft elke volgende diagnose een gok.
#
# GEBRUIK
#
#   thermal_guard.sh <unit> [max_c] [samples] [interval_s]
#
#   unit        systemd --user unit die gestopt wordt bij oververhitting
#   max_c       drempel in °C (default 78 — vier onder `high`, zes onder crit)
#   samples     zoveel opeenvolgende metingen boven de drempel vóór ingrijpen
#               (default 3; één uitschieter is ruis, drie is een trend)
#   interval_s  secondes tussen metingen (default 10)
# ═══════════════════════════════════════════════════════════════

set -uo pipefail

UNIT="${1:?geef de te bewaken unit}"
MAX_C="${2:-78}"
NEED="${3:-3}"
IVAL="${4:-10}"
LOG="${LOG:-/home/bart/CODE/docs/thermal_guard.log}"

# Schrijft naar TWEE plekken, en dat is geen luxe. Het vaste pad overleeft een
# aanroep zonder redirect; stdout zorgt dat het bewijs óók landt waar de
# operator het zoekt. Op 19-08-2026 draaide deze bewaker 206 metingen met een
# piek van 75 °C terwijl `-p StandardOutput=append:...` een leeg bestand
# opleverde, en de conclusie was "de guard doet niets" — precies de stille
# poort waartegen dit script is geschreven.
say() {
    local line
    line="$(date '+%F %T') $*"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$LOG"
    sync -d "$LOG" 2>/dev/null
}

pkg_temp() {
    # Package id 0 is de sensor waar `crit` op slaat. Val terug op de heetste
    # core als de package-sensor ontbreekt: te weinig meten is erger dan een
    # graad naast zitten.
    local t
    t=$(sensors -u 2>/dev/null | awk '/^Package id 0:/{f=1} f&&/temp[0-9]+_input:/{print int($2); exit}')
    [ -z "$t" ] && t=$(sensors -u 2>/dev/null | awk '/temp[0-9]+_input:/{if($2+0>m)m=$2+0} END{print int(m)}')
    echo "${t:-0}"
}

say "start: bewaakt ${UNIT}, drempel ${MAX_C}C, ${NEED} opeenvolgende metingen, elke ${IVAL}s"
say "log: ${LOG} (en stdout)"

n=0; peak=0; sum=0
over=0
while systemctl --user is-active --quiet "$UNIT"; do
    t=$(pkg_temp)
    load=$(awk '{print $1}' /proc/loadavg)
    memg=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
    mhz=$(awk '/cpu MHz/{s+=$4; n++} END{if(n)printf "%.0f", s/n}' /proc/cpuinfo)

    if [ "$t" -ge "$MAX_C" ]; then
        over=$((over + 1))
    else
        over=0
    fi
    n=$((n + 1)); sum=$((sum + t))
    [ "$t" -gt "$peak" ] && peak=$t
    say "temp=${t}C load=${load} mem_avail=${memg}G mhz=${mhz} over=${over}/${NEED}"

    if [ "$over" -ge "$NEED" ]; then
        say "STOP: ${t}C >= ${MAX_C}C bij ${NEED} opeenvolgende metingen — ${UNIT} wordt gestopt"
        systemctl --user stop "$UNIT"
        say "gestopt. De run is herstartbaar; de machine is dat niet zonder fysieke reset."
        exit 1
    fi
    sleep "$IVAL"
done

say "einde: ${UNIT} draait niet meer (klaar of gestopt)"
if [ "$n" -gt 0 ]; then
    say "samenvatting: ${n} metingen, piek ${peak}C, gemiddeld $((sum / n))C, drempel ${MAX_C}C"
else
    say "samenvatting: GEEN metingen — de unit draaide al niet meer bij de start"
fi
exit 0
