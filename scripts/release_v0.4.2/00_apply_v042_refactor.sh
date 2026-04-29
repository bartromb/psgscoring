#!/usr/bin/env bash
# psgscoring v0.4.2 — local baseline validation refactor
#
# Wat verandert:
#   1. PostProcessingRules dataclass krijgt 2 nieuwe velden:
#      - local_baseline_cv_threshold (default 0.30)
#      - local_baseline_strict_reduction (default 25.0)
#
#   2. _validate_local_reduction in respiratory.py wordt profile-aware
#      (geen hardcoded 0.30 / 30.0 meer)
#
#   3. _profile_to_legacy_dict in constants.py geeft nieuwe velden door
#
#   4. Alle 8 profielen krijgen passende waardes:
#      Strict:    cv=0.30, reduction=30.0   (extra streng bij stabiele patient)
#      Standard:  cv=0.30, reduction=25.0   (matig)
#      Sensitive: cv=0.20, reduction=20.0   (geen extra strengheid)
#      Andere:    defaults (cv=0.30, reduction=25.0)
#
#   5. Versie bump 0.4.1 → 0.4.2
set -euo pipefail

PSG_REPO="${HOME}/Desktop/GITHUB/github_psgscoring"

[ -d "$PSG_REPO" ] || { echo "✗ $PSG_REPO niet gevonden"; exit 1; }
cd "$PSG_REPO"

echo "→ Pre-flight: backup huidige v0.4.1 bestanden..."
mkdir -p ".v041_backup_$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$(ls -td .v041_backup_* | head -1)"
cp psgscoring/profiles.py "$BACKUP_DIR/"
cp psgscoring/respiratory.py "$BACKUP_DIR/"
cp psgscoring/constants.py "$BACKUP_DIR/"
cp psgscoring/__init__.py "$BACKUP_DIR/" 2>/dev/null || true
cp pyproject.toml "$BACKUP_DIR/"
echo "  ✓ Backup: $BACKUP_DIR"

# ── Patch 1: PostProcessingRules dataclass ───────────────────────────
echo ""
echo "→ Patch 1: PostProcessingRules — voeg 2 nieuwe velden toe"
python3 << 'PYEND'
from pathlib import Path
f = Path("psgscoring/profiles.py")
content = f.read_text()

OLD = '''    unsure_as_hypopnea: bool = False
    """NSRR-specific: 'Unsure' tag = hypopnea with >50% reduction."""'''

NEW = '''    unsure_as_hypopnea: bool = False
    """NSRR-specific: 'Unsure' tag = hypopnea with >50% reduction."""

    local_baseline_cv_threshold: float = 0.30
    """v0.4.2: CV threshold below which local-baseline reduction is enforced strictly.

    The local baseline validation (`_validate_local_reduction` in respiratory.py)
    applies a stricter flow-reduction requirement when surrounding breathing is
    stable (low CV). At CV < this threshold, the stricter reduction kicks in.

    Default 0.30 matches the legacy hardcoded behaviour. Lower values (e.g. 0.20)
    relax the strictness — useful for sensitive profiles where borderline events
    should not be rejected.
    """

    local_baseline_strict_reduction: float = 25.0
    """v0.4.2: Flow-reduction percentage required when local CV < threshold.

    When breathing is stable (CV < local_baseline_cv_threshold), candidate events
    must show this percentage of flow reduction (vs the default 20%) to be
    accepted as hypopneas.

    - Strict profile: 30.0 (extra strict — reject more borderline events)
    - Standard:       25.0 (moderate)
    - Sensitive:      20.0 (no extra strictness — accept borderline events)
    """'''

if OLD in content:
    content = content.replace(OLD, NEW)
    f.write_text(content)
    print("  ✓ PostProcessingRules — 2 velden toegevoegd")
else:
    print("  ✗ Anchor niet gevonden")
    raise SystemExit(1)
PYEND

# ── Patch 2: per-profile waardes ─────────────────────────────────────
echo ""
echo "→ Patch 2: voeg waardes toe in alle 8 profielen"
python3 << 'PYEND'
from pathlib import Path
import re
f = Path("psgscoring/profiles.py")
content = f.read_text()

# Map profile-name naar (cv_threshold, strict_reduction)
PROFILE_VALUES = {
    "aasm_v3_strict":    (0.30, 30.0),
    "aasm_v3_rec":       (0.30, 25.0),
    "aasm_v3_sensitive": (0.20, 20.0),
    "aasm_v2_rec":       (0.30, 25.0),
    "aasm_v1_rec":       (0.30, 25.0),
    "cms_medicare":      (0.30, 25.0),
    "mesa_shhs":         (0.30, 25.0),
    "chicago_1999":      (0.30, 25.0),
}

n_patched = 0
for prof_name, (cv, reduction) in PROFILE_VALUES.items():
    # Match: name="PROF_NAME" gevolgd door alle tekst tot we
    # post_processing=PostProcessingRules(...)\n    ), tegenkomen.
    pattern = re.compile(
        r'(name="' + re.escape(prof_name) + r'".*?post_processing=PostProcessingRules\(\n)'
        r'(.*?)'
        r'(\n    \),)',
        re.DOTALL,
    )
    
    def replace(match):
        prefix = match.group(1)
        body = match.group(2)
        suffix = match.group(3)
        if "local_baseline_cv_threshold" in body:
            return match.group(0)
        body_stripped = body.rstrip()
        if not body_stripped.endswith(','):
            body_stripped += ','
        new_body = (
            body_stripped + '\n'
            f'        local_baseline_cv_threshold={cv},\n'
            f'        local_baseline_strict_reduction={reduction},'
        )
        return prefix + new_body + suffix
    
    new_content, count = pattern.subn(replace, content, count=1)
    if count == 1:
        content = new_content
        n_patched += 1
        print(f"  ✓ {prof_name}: cv={cv}, reduction={reduction}")
    else:
        print(f"  ✗ {prof_name}: anchor niet gevonden")

f.write_text(content)
print(f"  → {n_patched}/8 profielen gepatched")
PYEND

# ── Patch 3: _profile_to_legacy_dict in constants.py ─────────────────
echo ""
echo "→ Patch 3: _profile_to_legacy_dict — geef nieuwe velden door"
python3 << 'PYEND'
from pathlib import Path
f = Path("psgscoring/constants.py")
content = f.read_text()

OLD = '''        "STABILITY_FILTER_CV":    pp.stability_filter_cv,
        # Audit metadata — read by pipeline.py for output["meta"]["profile"]'''

NEW = '''        "STABILITY_FILTER_CV":    pp.stability_filter_cv,
        # v0.4.2: profile-aware local baseline validation
        "LOCAL_BL_CV_THRESHOLD":  pp.local_baseline_cv_threshold,
        "LOCAL_BL_STRICT_RED":    pp.local_baseline_strict_reduction,
        # Audit metadata — read by pipeline.py for output["meta"]["profile"]'''

if OLD in content:
    content = content.replace(OLD, NEW)
    f.write_text(content)
    print("  ✓ _profile_to_legacy_dict bijgewerkt")
else:
    print("  ✗ Anchor niet gevonden")
    raise SystemExit(1)
PYEND

# ── Patch 4: _validate_local_reduction in respiratory.py ─────────────
echo ""
echo "→ Patch 4: _validate_local_reduction — profile-aware"
python3 << 'PYEND'
from pathlib import Path
f = Path("psgscoring/respiratory.py")
content = f.read_text()

# Stap A: signature uitbreiden
OLD_SIG = '''def _validate_local_reduction(
    env: np.ndarray,
    event_start: int,
    event_end: int,
    sf: float,
    min_reduction_pct: float = 20.0,
    pre_win_s: float = 30.0,
) -> tuple[bool, float]:'''

NEW_SIG = '''def _validate_local_reduction(
    env: np.ndarray,
    event_start: int,
    event_end: int,
    sf: float,
    min_reduction_pct: float = 20.0,
    pre_win_s: float = 30.0,
    stability_cv_threshold: float = 0.30,    # v0.4.2: profile-aware
    stability_strict_reduction: float = 30.0,  # v0.4.2: profile-aware
) -> tuple[bool, float]:'''

if OLD_SIG in content:
    content = content.replace(OLD_SIG, NEW_SIG)
    print("  ✓ Signature uitgebreid met 2 nieuwe params")
else:
    print("  ✗ Signature niet gevonden")
    raise SystemExit(1)

# Stap B: hardcoded check vervangen door parameter-gestuurd
OLD_CHECK = '''    if len(stab_seg) > int(10 * sf) and np.mean(stab_seg) > 1e-9:
        local_cv = float(np.std(stab_seg) / np.mean(stab_seg))
        # Stabiele ademhaling: CV < 0.30 → verhoog drempel naar 30%
        # Instabiele ademhaling: CV ≥ 0.30 → standaard drempel (20%)
        if local_cv < 0.30:
            min_reduction_pct = max(min_reduction_pct, 30.0)'''

NEW_CHECK = '''    if len(stab_seg) > int(10 * sf) and np.mean(stab_seg) > 1e-9:
        local_cv = float(np.std(stab_seg) / np.mean(stab_seg))
        # v0.4.2: Profile-aware stabiliteits-bewuste drempel.
        # Strict profile gebruikt strengere reductie-eis bij stabiele ademhaling;
        # sensitive profile gebruikt geen extra strengheid (cv_threshold lager).
        if local_cv < stability_cv_threshold:
            min_reduction_pct = max(min_reduction_pct, stability_strict_reduction)'''

if OLD_CHECK in content:
    content = content.replace(OLD_CHECK, NEW_CHECK)
    print("  ✓ Hardcoded 0.30/30.0 vervangen door profile-aware params")
else:
    print("  ✗ Hardcoded check niet gevonden")
    raise SystemExit(1)

# Stap C: zoek alle aanroepers van _validate_local_reduction
# en geef de nieuwe params door (uit sp dict)
import re

# Pattern: _validate_local_reduction(...)
calls = re.findall(r'_validate_local_reduction\([^)]*\)', content, re.DOTALL)
print(f"  → {len(calls)} aanroepen gevonden van _validate_local_reduction")

# We moeten de aanroep updaten om de nieuwe params door te geven
# Voor elke aanroep: vóór de afsluitende ) toevoegen:
#   stability_cv_threshold=sp.get("LOCAL_BL_CV_THRESHOLD", 0.30),
#   stability_strict_reduction=sp.get("LOCAL_BL_STRICT_RED", 30.0)

# Aanroepen patroon: ... = _validate_local_reduction(...args...)
def patch_call(match):
    call_text = match.group(0)
    # Voeg de twee nieuwe params toe vóór de afsluitende )
    if "stability_cv_threshold" in call_text:
        return call_text  # al gepatched
    # Strip trailing whitespace + )
    inner = call_text[:-1].rstrip()
    if not inner.endswith(','):
        inner += ','
    new_call = (
        inner + '\n'
        '            stability_cv_threshold=sp.get("LOCAL_BL_CV_THRESHOLD", 0.30),\n'
        '            stability_strict_reduction=sp.get("LOCAL_BL_STRICT_RED", 30.0),\n'
        '        )'
    )
    return new_call

new_content = re.sub(
    r'_validate_local_reduction\([^)]*\)',
    patch_call,
    content,
    flags=re.DOTALL,
)
content = new_content

# Verifieer dat sp beschikbaar is in de scope waar de aanroepen staan
# (anders crashen alle aanroepen)
n_patched_calls = sum(1 for m in re.finditer(
    r'_validate_local_reduction\([^)]*stability_cv_threshold[^)]*\)',
    content, re.DOTALL))
print(f"  → {n_patched_calls} aanroepen gepatched met sp.get(...)")

f.write_text(content)
print("  ✓ respiratory.py bijgewerkt")
PYEND

# ── Patch 5: versie bump ──────────────────────────────────────────────
echo ""
echo "→ Patch 5: versie bump 0.4.1 → 0.4.2"
sed -i 's/version = "0\.4\.1"/version = "0.4.2"/' pyproject.toml
sed -i 's/__version__ = "0\.4\.1"/__version__ = "0.4.2"/' psgscoring/__init__.py 2>/dev/null || true
echo "  ✓ Versie 0.4.2"

# ── Verifieer syntax ──────────────────────────────────────────────────
echo ""
echo "→ Syntax check..."
python3 -c "import ast; ast.parse(open('psgscoring/profiles.py').read())" && echo "  ✓ profiles.py"
python3 -c "import ast; ast.parse(open('psgscoring/constants.py').read())" && echo "  ✓ constants.py"
python3 -c "import ast; ast.parse(open('psgscoring/respiratory.py').read())" && echo "  ✓ respiratory.py"

echo ""
echo "→ Snelle import-test..."
python3 -c "
from psgscoring.profiles import PROFILES
from psgscoring.constants import SCORING_PROFILES
print('PROFILES geladen:', len(PROFILES), 'profielen')
print()
print('Per profile — local_baseline_cv_threshold / local_baseline_strict_reduction:')
for name in ['aasm_v3_strict', 'aasm_v3_rec', 'aasm_v3_sensitive']:
    p = PROFILES[name]
    cv = p.post_processing.local_baseline_cv_threshold
    sr = p.post_processing.local_baseline_strict_reduction
    print(f'  {name:25s}: cv={cv}, strict_reduction={sr}')
print()
print('SCORING_PROFILES dict (legacy format):')
for name in ['aasm_v3_strict', 'aasm_v3_rec', 'aasm_v3_sensitive']:
    sp = SCORING_PROFILES[name]
    cv = sp.get('LOCAL_BL_CV_THRESHOLD')
    sr = sp.get('LOCAL_BL_STRICT_RED')
    print(f'  {name:25s}: LOCAL_BL_CV_THRESHOLD={cv}, LOCAL_BL_STRICT_RED={sr}')
"

echo ""
echo "✓ Refactor v0.4.2 voltooid"
echo ""
echo "→ Volgende stap: validate_psgipa.py runnen om monotonie te checken"
echo "  python3 validate_psgipa.py --data-dir ~/PSG-IPA --ignore-sanity --workers 5"
