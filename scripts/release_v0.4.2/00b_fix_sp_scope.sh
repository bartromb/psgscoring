#!/usr/bin/env bash
# v0.4.2 fix — repareer de "name 'sp' is not defined" fout
#
# Probleem: mijn vorige patch zette sp.get(...) calls in _detect_hypopneas
#           waar sp niet bestaat (sp leeft alleen in detect_respiratory_events)
#
# Fix: 
#   1. Voeg 2 nieuwe parameters toe aan _detect_hypopneas signature
#   2. Geef ze door op de aanroep-site (regel 533) waar sp wel bestaat
#   3. De aanroep van _validate_local_reduction binnen _detect_hypopneas
#      gebruikt dan de doorgegeven parameters
set -euo pipefail

PSG_REPO="${HOME}/Desktop/GITHUB/github_psgscoring"
[ -d "$PSG_REPO" ] || { echo "✗ $PSG_REPO niet gevonden"; exit 1; }
cd "$PSG_REPO"

echo "→ Fix sp-scope bug in _detect_hypopneas..."

python3 << 'PYEND'
from pathlib import Path
import re
f = Path("psgscoring/respiratory.py")
content = f.read_text()

# ── Stap 1: voeg 2 nieuwe parameters toe aan _detect_hypopneas signature ──
OLD_SIG = '''def _detect_hypopneas(
    hypopnea_raw, sleep_mask_hy, hypop_env, hypop_norm, hypop_baseline,
    sf_hy, sf_flow, sf_spo2, hypno,
    thorax_env, abdomen_env, thorax_raw, abdomen_raw, effort_bl,
    spo2_data, global_spo2_bl, existing_events,
    apply_spo2_crosscontam_fix: bool = True,
    desat_pct: float = 3.0,
    contam_win_s: float = 15.0,
    post_event_win_s: float = 45,
    max_dur_s: float = HYPOPNEA_MAX_DUR_S,
    ecg_data=None, tecg=None, r_peaks=None, sf_ecg=None,
    flow_filt: np.ndarray | None = None,
    breaths: list | None = None,
    signal_quality: dict | None = None,
) -> tuple[list[dict], list[dict]]:'''

NEW_SIG = '''def _detect_hypopneas(
    hypopnea_raw, sleep_mask_hy, hypop_env, hypop_norm, hypop_baseline,
    sf_hy, sf_flow, sf_spo2, hypno,
    thorax_env, abdomen_env, thorax_raw, abdomen_raw, effort_bl,
    spo2_data, global_spo2_bl, existing_events,
    apply_spo2_crosscontam_fix: bool = True,
    desat_pct: float = 3.0,
    contam_win_s: float = 15.0,
    post_event_win_s: float = 45,
    max_dur_s: float = HYPOPNEA_MAX_DUR_S,
    ecg_data=None, tecg=None, r_peaks=None, sf_ecg=None,
    flow_filt: np.ndarray | None = None,
    breaths: list | None = None,
    signal_quality: dict | None = None,
    local_bl_cv_threshold: float = 0.30,        # v0.4.2: profile-aware
    local_bl_strict_reduction: float = 30.0,    # v0.4.2: profile-aware
) -> tuple[list[dict], list[dict]]:'''

if OLD_SIG in content:
    content = content.replace(OLD_SIG, NEW_SIG)
    print("  ✓ _detect_hypopneas signature uitgebreid met 2 nieuwe params")
else:
    print("  ✗ Signature niet gevonden")
    raise SystemExit(1)

# ── Stap 2: vervang sp.get(...) in _validate_local_reduction call door params ──
# Gevonden patroon na vorige patch:
OLD_CALL = '''            local_valid, local_red = _validate_local_reduction(
                hypop_env, sub_idx[0], sub_idx[-1] + 1, sf_hy,
            stability_cv_threshold=sp.get("LOCAL_BL_CV_THRESHOLD", 0.30),
            stability_strict_reduction=sp.get("LOCAL_BL_STRICT_RED", 30.0),
        )'''

NEW_CALL = '''            local_valid, local_red = _validate_local_reduction(
                hypop_env, sub_idx[0], sub_idx[-1] + 1, sf_hy,
                stability_cv_threshold=local_bl_cv_threshold,
                stability_strict_reduction=local_bl_strict_reduction,
            )'''

if OLD_CALL in content:
    content = content.replace(OLD_CALL, NEW_CALL)
    print("  ✓ Call van _validate_local_reduction gebruikt nu function params")
else:
    # Fallback: probeer een meer flexibele match
    pattern = re.compile(
        r'local_valid, local_red = _validate_local_reduction\(\s*'
        r'hypop_env,\s*sub_idx\[0\],\s*sub_idx\[-1\] \+ 1,\s*sf_hy,\s*'
        r'\s*stability_cv_threshold=sp\.get\([^)]*\),\s*'
        r'\s*stability_strict_reduction=sp\.get\([^)]*\),\s*\)',
        re.DOTALL,
    )
    REPLACE = '''local_valid, local_red = _validate_local_reduction(
                hypop_env, sub_idx[0], sub_idx[-1] + 1, sf_hy,
                stability_cv_threshold=local_bl_cv_threshold,
                stability_strict_reduction=local_bl_strict_reduction,
            )'''
    new_content, count = pattern.subn(REPLACE, content)
    if count == 1:
        content = new_content
        print(f"  ✓ Call gepatched via flexibele regex")
    else:
        print(f"  ✗ Geen match gevonden — handmatige inspectie nodig")
        # Toon huidige call voor diagnose
        m = re.search(r'_validate_local_reduction\([^)]*\)', content, re.DOTALL)
        if m:
            print(f"  Huidige call:\n{m.group(0)}")
        raise SystemExit(1)

# ── Stap 3: pas de aanroep van _detect_hypopneas aan om de 2 nieuwe params door te geven ──
# Locate the call site (regel 533 area)
OLD_CALL_SITE = '''        new_events, rejected = _detect_hypopneas(
            hypopnea_raw_corrected, sleep_mask_hy, hypop_env,
            hypop_norm_corrected, hypop_baseline_corrected,
            sf_hy, sf_flow, sf_spo2, hypno,
            thorax_env, abdomen_env, thorax_data, abdomen_data, effort_bl,
            spo2_data, global_spo2_bl, events,'''

# We moeten de call-site finden en achteraan de twee nieuwe kwargs toevoegen
# voordat de afsluitende ) komt.

# Pattern: vind "new_events, rejected = _detect_hypopneas(" en alles tot afsluitende )
pattern_call = re.compile(
    r'(new_events, rejected = _detect_hypopneas\(\n)(.*?)(\n        \))',
    re.DOTALL,
)

def patch_call(match):
    prefix = match.group(1)
    body = match.group(2)
    suffix = match.group(3)
    
    if "local_bl_cv_threshold" in body:
        return match.group(0)  # idempotent
    
    body_stripped = body.rstrip()
    if not body_stripped.endswith(','):
        body_stripped += ','
    new_body = (
        body_stripped + '\n'
        '            local_bl_cv_threshold=sp.get("LOCAL_BL_CV_THRESHOLD", 0.30),\n'
        '            local_bl_strict_reduction=sp.get("LOCAL_BL_STRICT_RED", 30.0),'
    )
    return prefix + new_body + suffix

new_content, count = pattern_call.subn(patch_call, content)
if count == 1:
    content = new_content
    print(f"  ✓ _detect_hypopneas call site aangepast: 2 sp.get(...) kwargs toegevoegd")
elif count == 0:
    print(f"  ✗ Call site niet gevonden")
    raise SystemExit(1)
else:
    print(f"  ⚠ Meer dan 1 match ({count}) — controleer output")

# ── Verifieer syntax ──────────────────────────────────────────────────
import ast
try:
    ast.parse(content)
    f.write_text(content)
    print("  ✓ Syntax OK, bestand opgeslagen")
except SyntaxError as e:
    print(f"  ✗ Syntax error: {e}")
    raise SystemExit(1)

PYEND

# ── Snelle smoke test ─────────────────────────────────────────────────
echo ""
echo "→ Smoke test: kan psgscoring importeren?"
python3 -c "
from psgscoring.respiratory import _detect_hypopneas, _validate_local_reduction
import inspect
sig = inspect.signature(_detect_hypopneas)
params = list(sig.parameters.keys())
assert 'local_bl_cv_threshold' in params, 'Parameter ontbreekt!'
assert 'local_bl_strict_reduction' in params, 'Parameter ontbreekt!'
print(f'  ✓ _detect_hypopneas heeft local_bl_* params')
print(f'  Total params: {len(params)}')
"

echo ""
echo "✓ Fix toegepast"
echo ""
echo "→ Run nu opnieuw:"
echo "  python3 validate_psgipa.py --data-dir ~/PSG-IPA --ignore-sanity --workers 5"
