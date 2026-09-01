"""
psgscoring.classify
===================
AASM apnea-type classification: obstructive / central / mixed.

v0.8.11 additions
-----------------
- Phase angle (Hilbert transform) on thorax/abdomen instantaneous phase →
  continuous asynchrony in degrees. >45° during flow limitation → obstructive
  with high confidence, largely eliminating Rule-6 borderline defaults.
- LightGBM confidence calibration (optional): if a pre-trained model is
  available at ``LGBM_MODEL_PATH``, per-event features are fed to the model
  for a data-driven confidence score (0–1). Falls back to rule-based when the
  model is unavailable.

Dependencies: numpy, scipy, psgscoring.constants, psgscoring.utils
Optional:     lightgbm (confidence calibration only)
"""

from __future__ import annotations
import logging
import os
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import hilbert

from .constants import EFFORT_ABSENT_RATIO, EFFORT_PRESENT_RATIO
from .utils import safe_r

logger = logging.getLogger("psgscoring.classify")

# ---------------------------------------------------------------------------
# LightGBM confidence calibration (optional)
# ---------------------------------------------------------------------------

# Set PSGSCORING_LGBM_MODEL to an absolute path to enable calibration.
# The model must accept 10 features in this order (see _extract_lgbm_features).
LGBM_MODEL_PATH: str | None = os.environ.get("PSGSCORING_LGBM_MODEL", None)
_lgbm_model = None   # loaded on first use


def _load_lgbm_model():
    """Load LightGBM model once, cache in module-level variable."""
    global _lgbm_model
    if _lgbm_model is not None or LGBM_MODEL_PATH is None:
        return _lgbm_model
    try:
        import lightgbm as lgb
        _lgbm_model = lgb.Booster(model_file=LGBM_MODEL_PATH)
        logger.info("LightGBM confidence model loaded from %s", LGBM_MODEL_PATH)
    except Exception as e:
        logger.warning("LightGBM model load failed (%s) — using rule-based confidence", e)
        _lgbm_model = None
    return _lgbm_model


def _extract_lgbm_features(
    effort_ratio: float,
    raw_var_ratio: float,
    paradox_corr: float | None,
    first_ratio: float,
    second_ratio: float,
    quarter_efforts: list[float],
    phase_angle_deg: float | None,
    duration_s: float,
    rule_idx: int,
) -> list[float]:
    """
    Extract 10 numeric features for the LightGBM confidence model.

    Feature order must match the training schema exactly.
    """
    return [
        float(effort_ratio),
        float(raw_var_ratio),
        float(paradox_corr) if paradox_corr is not None else 0.0,
        float(first_ratio),
        float(second_ratio),
        float(np.mean(quarter_efforts)) if quarter_efforts else 0.0,
        float(np.std(quarter_efforts))  if quarter_efforts else 0.0,
        float(phase_angle_deg) if phase_angle_deg is not None else 0.0,
        float(duration_s),
        float(rule_idx),   # which rule fired (1–6)
    ]


def _lgbm_confidence(features: list[float]) -> float | None:
    """Return LightGBM confidence or None if model unavailable."""
    model = _load_lgbm_model()
    if model is None:
        return None
    try:
        import numpy as np
        X = np.array([features], dtype=np.float32)
        pred = model.predict(X)
        return float(np.clip(pred[0], 0.0, 1.0))
    except Exception as e:
        logger.debug("LightGBM predict failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main classification entry point
# ---------------------------------------------------------------------------

def classify_apnea_type(
    onset_idx: int,
    end_idx: int,
    thorax_env:   np.ndarray | None,
    abdomen_env:  np.ndarray | None,
    thorax_raw:   np.ndarray | None,
    abdomen_raw:  np.ndarray | None,
    effort_baseline: float,
    sf: float,
    ecg_assessment: dict | None = None,
    flattening_index: float | None = None,
    signal_quality: dict | None = None,
    use_rhythm: bool = False,
    phase_angle_needs_effort: bool = False,
) -> tuple[str, float, dict]:
    """
    Classify an apnea event as ``"obstructive"``, ``"central"``, or
    ``"mixed"`` per AASM Adult Scoring Rules (section 3B).

    Decision logic (in priority order)
    -----------------------------------
    0. **Phase angle >45°** (v0.8.11) → obstructive with high confidence.
       Detects paradoxical movement even before amplitude drops.
    1. **Paradoxical thoraco-abdominal movement** → obstructive
    2. **Raw signal variability present, envelope low** → obstructive
    3. **Mixed pattern** – absent effort first half, present second half
    4. **Effort clearly present** → obstructive
    5. **Truly flat** – no raw movement, low envelope → central
    5b. **ECG-derived reclassification** (v0.8.23) – TECG + spectral
        analysis overrides RIP-based obstructive if cardiac artefact only.
    6. **Borderline default** → obstructive (low confidence)
       If LightGBM model available, confidence is calibrated by model.

    Parameters
    ----------
    ecg_assessment : dict, optional
        Output of ``ecg_effort.ecg_effort_assessment()``.
        If provided and ``reclassify_as_central`` is True, events that
        would otherwise be classified as obstructive (rules 4, 6) are
        reclassified as central.
    flattening_index : float, optional
        Mean inspiratory flattening index for the event (0–1).
        >0.30 indicates flow limitation (supports obstructive);
        <0.10 with low effort supports central classification.
        Computed by ``breath.compute_flattening_index()``.

    Returns
    -------
    (type_str, confidence_0_to_1, detail_dict)
    """
    # ── Rule -1 (v0.3.001 BUG2 MARKER): RIP pair quality gate ───────────
    # Respects compare_rip_pair() output. Single-channel failures cannot
    # be classified via bilateral effort analysis (Rules 0-6 all depend
    # on trustworthy thorax+abdomen signals). Route to fallback BEFORE
    # attempting the bilateral chain.
    #
    # Clinical motivation: Loos case (AZORG April 2026). Thorax RIP dead
    # (energy ratio 6861x). Without this gate, classifier defaults to
    # obstructive because bilateral analysis sees "no paradox" - but
    # that's because the signal is absent, not because there was no
    # paradoxical movement.
    if signal_quality is not None:
        _mode = signal_quality.get("recommended_mode")

        if _mode == "single-channel":
            from .signal_quality import single_channel_fallback_classify

            _working = signal_quality.get("working_channel")
            _working_raw = None
            if _working == "thorax":
                _working_raw = thorax_raw
            elif _working == "abdomen":
                _working_raw = abdomen_raw

            if _working_raw is None:
                return "uncertain", 0.3, {
                    "classification_source": "single-channel-no-signal",
                    "decision_reason": (
                        f"gate={_mode} but working_channel={_working!r} "
                        f"signal is None"
                    ),
                }

            _start_s = onset_idx / max(sf, 1)
            _end_s = end_idx / max(sf, 1)
            _fallback_type = single_channel_fallback_classify(
                apnea_start_s=_start_s,
                apnea_end_s=_end_s,
                effort_signal=_working_raw,
                sf=sf,
                use_rhythm=use_rhythm,
            )

            if _fallback_type == "uncertain":
                _conf = 0.4
            elif _fallback_type == "central":
                _conf = 0.65
            else:
                _conf = 0.55

            return _fallback_type, _conf, {
                "classification_source": "single-channel-fallback",
                "working_channel": _working,
                "energy_ratio": signal_quality.get("energy_ratio"),
                "decision_reason": (
                    f"rip_pair_gate=single-channel working={_working}"
                ),
            }

        elif _mode == "unreliable":
            return "uncertain", 0.2, {
                "classification_source": "unreliable-rip-pair",
                "decision_reason": "rip_pair_gate=unreliable",
                "energy_ratio": signal_quality.get("energy_ratio"),
            }

        # mode == "bilateral" or unrecognised → fall through to Rules 0-6

    seg_len  = end_idx - onset_idx
    dur_s    = seg_len / max(sf, 1)
    if seg_len < 2:
        return "obstructive", 0.5, {}

    effort_segs: dict[str, np.ndarray] = {}
    if thorax_env is not None:
        effort_segs["thorax"]  = thorax_env[onset_idx:end_idx]
    if abdomen_env is not None:
        effort_segs["abdomen"] = abdomen_env[onset_idx:end_idx]

    if not effort_segs:
        return "obstructive", 0.3, {"note": "no effort channels"}

    event_effort  = float(np.mean([np.mean(s) for s in effort_segs.values()]))
    effort_ratio  = event_effort / effort_baseline if effort_baseline > 1e-9 else 0.0
    raw_var_ratio = _compute_raw_variability(thorax_raw, abdomen_raw, onset_idx, end_idx, sf)
    paradox_corr  = _compute_paradox_correlation(thorax_raw, abdomen_raw, onset_idx, end_idx)

    # v0.8.11: Phase angle via Hilbert transform
    phase_angle_deg = _compute_phase_angle(thorax_raw, abdomen_raw, onset_idx, end_idx, sf)

    half         = seg_len // 2
    first_ratio  = _mean_effort_ratio(effort_segs, 0, half, effort_baseline)
    second_ratio = _mean_effort_ratio(effort_segs, half, seg_len, effort_baseline)
    quarter      = max(1, seg_len // 4)
    quarter_efforts = [
        _mean_effort_ratio(effort_segs, q * quarter,
                           min((q + 1) * quarter, seg_len), effort_baseline)
        for q in range(4)
    ]

    detail = {
        "effort_ratio":        safe_r(effort_ratio,    3),
        "raw_var_ratio":       safe_r(raw_var_ratio,   3),
        "first_half_effort":   safe_r(first_ratio,     3),
        "second_half_effort":  safe_r(second_ratio,    3),
        "quarter_efforts":     [safe_r(q, 3) for q in quarter_efforts],
        "paradox_correlation": safe_r(paradox_corr,    3),
        "phase_angle_deg":     safe_r(phase_angle_deg, 1),
        "flattening_index":    safe_r(flattening_index, 3),
    }

    is_paradox   = paradox_corr is not None and paradox_corr < -0.15
    has_raw_move = raw_var_ratio > 0.25

    # v0.2.5: Flattening index modulates confidence
    # High flattening (>0.30) = flow limitation = obstructive evidence
    # Low flattening (<0.10) with low effort = supports central
    _flat_obstr_boost = 0.0
    _flat_central_boost = 0.0
    if flattening_index is not None:
        if flattening_index > 0.30:
            _flat_obstr_boost = min(0.10, (flattening_index - 0.30) * 0.25)
        elif flattening_index < 0.10 and effort_ratio < EFFORT_ABSENT_RATIO:
            _flat_central_boost = min(0.10, (0.10 - flattening_index) * 0.5)

    # Helper to optionally replace rule-based confidence with LightGBM
    def _conf(rule_conf: float, rule_idx: int) -> float:
        """Bereken betrouwbaarheidsscore voor apnea-classificatie (0–1)."""
        features = _extract_lgbm_features(
            effort_ratio, raw_var_ratio, paradox_corr,
            first_ratio, second_ratio, quarter_efforts,
            phase_angle_deg, dur_s, rule_idx,
        )
        lgbm_c = _lgbm_confidence(features)
        if lgbm_c is not None:
            detail["lgbm_confidence"] = safe_r(lgbm_c, 3)
        return lgbm_c if lgbm_c is not None else rule_conf

    # ── Rule 0 (v0.8.11): Phase angle ≥45° during event ──────────────────
    #
    # MET AMPLITUDEPOORT (v0.31.6, `phase_angle_needs_effort`). Deze regel
    # vuurt vóór alle andere en kende geen ondergrens op de amplitude.
    # `_compute_phase_angle` is er expliciet op ontworpen om ook te werken
    # "wanneer de amplitude-envelop laag is" -- zinnig om een obstructief event
    # met lage amplitude te vangen, en precies verkeerd wanneer die lage
    # amplitude JUIST het centrale kenmerk is.
    #
    # Bij een centrale apneu bewegen thorax en abdomen per definitie
    # nauwelijks; de Hilbert-fase van twee bijna-vlakke signalen is die van
    # ruis, en ruis is niet in fase.
    #
    # Gemeten op PSG-IPA (5 opnames): van de 75 apneus die de scoorder
    # centraal noemde, noemden wij er 60 obstructief -- en 33 daarvan op
    # precies deze regel. De omgekeerde richting klopte bijna perfect (153 van
    # 154), dus dit is geen classificatiefout maar een eenzijdige bias.
    # De poort dekt ELKE paradox-afgeleide aanwijzing, niet alleen de
    # fasehoek. Regel 1 gebruikt de paradoxCORRELATIE, en die heeft exact
    # hetzelfde probleem: de correlatie tussen twee ruissignalen in tegenfase
    # is -1, ongeacht of er ademhaling onder zit. Alleen regel 0 dichtzetten
    # verplaatst de fout één regel naar beneden -- gemeten: hetzelfde event
    # kwam er dan uit op `paradox_corr=-0.997`.
    _vormmaten_genegeerd = (phase_angle_needs_effort
                       and effort_ratio < EFFORT_ABSENT_RATIO)
    if _vormmaten_genegeerd:
        # DRIE regels lezen hier hetzelfde ruissignaal als obstructie, en ze
        # moeten alle drie mee. Gemeten door de poort stapsgewijs te verbreden
        # op dezelfde fixture:
        #
        #   alleen regel 0 gepoort  -> regel 1 vuurt: paradox_corr=-0,997
        #   ook regel 1 gepoort     -> regel 2 vuurt: raw_movement_var=0,965
        #
        # Het gemeenschappelijke: fase, correlatie en variabiliteit zijn
        # VORMmaten. Ze zeggen iets over de structuur van een signaal, niets
        # over of er signaal IS. Onder de effortdrempel meten ze ruis, en ruis
        # heeft vorm.
        #
        # Wat overblijft is dat afwezige inspanning zelf beslist -- wat de
        # AASM-definitie van een centrale apneu ook zegt.
        detail["phase_angle_ignored"] = True
        is_paradox = False
        has_raw_move = False
    if (phase_angle_deg is not None and phase_angle_deg >= 45.0
            and not _vormmaten_genegeerd):
        conf = min(0.97, 0.75 + (phase_angle_deg - 45) / 180 * 0.2 + _flat_obstr_boost)
        detail["decision_reason"] = f"phase_angle={safe_r(phase_angle_deg,1)}deg"
        return "obstructive", safe_r(_conf(conf, 0), 2), detail

    # ── Rule 1: Paradox + raw movement ────────────────────────────────────
    if is_paradox and has_raw_move:
        conf = min(0.95, 0.70 + abs(paradox_corr) * 0.3)
        detail["decision_reason"] = f"paradox_corr={safe_r(paradox_corr,3)}"
        return "obstructive", safe_r(_conf(conf, 1), 2), detail

    # ── Rule 2: Raw movement, low envelope ───────────────────────────────
    if (raw_var_ratio > 0.40 and effort_ratio < EFFORT_PRESENT_RATIO
            and not _vormmaten_genegeerd):
        if paradox_corr is None or paradox_corr < 0.3:
            conf = min(0.85, 0.50 + raw_var_ratio * 0.3)
            detail["decision_reason"] = f"raw_movement_var={safe_r(raw_var_ratio,3)}"
            return "obstructive", safe_r(_conf(conf, 2), 2), detail

    # ── Rule 3: Mixed ─────────────────────────────────────────────────────
    # v0.8.30: relaxed first-half threshold (0.20 → 0.35) to catch mixed
    # apneas with gradual effort onset (not always a clean binary transition)
    if first_ratio < 0.35 and second_ratio > EFFORT_PRESENT_RATIO:
        # Stronger mixed signal when first half is truly absent
        mixed_conf = 0.6 + (second_ratio - first_ratio) * 0.5
        if first_ratio < EFFORT_ABSENT_RATIO:
            mixed_conf += 0.15  # classic mixed: absent → present
        conf = min(0.95, mixed_conf)
        detail["decision_reason"] = (
            f"mixed_first={safe_r(first_ratio,3)}_second={safe_r(second_ratio,3)}"
        )
        return "mixed", safe_r(_conf(conf, 3), 2), detail

    # ── Rule 4: Clear effort ──────────────────────────────────────────────
    if effort_ratio > EFFORT_PRESENT_RATIO:
        conf = min(0.95, 0.5 + (effort_ratio - EFFORT_PRESENT_RATIO))
        detail["decision_reason"] = f"effort_present={safe_r(effort_ratio,3)}"
        return "obstructive", safe_r(_conf(conf, 4), 2), detail

    # ── Rule 5: Truly flat → central ─────────────────────────────────────
    # v0.8.30: relaxed thresholds to account for cardiac pulsation artefact
    # on RIP bands (typically raw_var 0.10–0.20, effort_ratio 0.10–0.25)
    quarters_absent = sum(1 for q in quarter_efforts if q < EFFORT_ABSENT_RATIO)
    quarters_low    = sum(1 for q in quarter_efforts if q < EFFORT_PRESENT_RATIO)
    # De vormmaten tellen hier NEUTRAAL wanneer de poort dicht staat. Anders
    # blokkeert dezelfde ruis die geen obstructie meer mag aantonen, wél nog
    # de centrale regel: `no_paradox` en `raw_var < 0,25` zijn óók vormmaten.
    # Zonder deze regel valt een event met afwezige inspanning door naar de
    # restcategorie -- de tweede foutbron uit de PSG-IPA-meting (27 van 60).
    no_paradox      = (_vormmaten_genegeerd or paradox_corr is None
                       or paradox_corr > -0.10)
    no_phase_signal = (_vormmaten_genegeerd or phase_angle_deg is None
                       or phase_angle_deg < 30.0)
    _raw_var_beslis = 0.0 if _vormmaten_genegeerd else raw_var_ratio
    if (
        _raw_var_beslis < 0.25 and
        effort_ratio  < EFFORT_ABSENT_RATIO and
        quarters_absent >= 2 and
        no_paradox and
        no_phase_signal
    ):
        conf = min(0.90, 0.5 + (EFFORT_ABSENT_RATIO - effort_ratio) * 3 + _flat_central_boost)
        detail["decision_reason"] = (
            f"truly_flat_var={safe_r(raw_var_ratio,3)}_effort={safe_r(effort_ratio,3)}"
        )
        return "central", safe_r(_conf(conf, 5), 2), detail

    # ── Rule 5a (v0.8.30): Probable central — low effort, no paradox ─────
    # Catches events where effort is low but not fully absent (cardiac
    # pulsation artefact inflates effort_ratio to 0.20–0.35).
    if (
        _raw_var_beslis < 0.30 and
        effort_ratio  < EFFORT_PRESENT_RATIO and   # < 0.40
        quarters_low  >= 3 and                      # most quarters below 0.40
        no_paradox and
        no_phase_signal and
        not is_paradox and
        not has_raw_move
    ):
        conf = min(0.75, 0.45 + (EFFORT_PRESENT_RATIO - effort_ratio) + _flat_central_boost)
        detail["decision_reason"] = (
            f"probable_central_var={safe_r(raw_var_ratio,3)}_effort={safe_r(effort_ratio,3)}"
        )
        return "central", safe_r(_conf(conf, 5), 2), detail

    # ── Rule 5b (v0.8.23): ECG-derived reclassification ──────────────────
    # If TECG shows no inspiratory bursts AND spectral analysis shows
    # cardiac dominance, reclassify borderline/effort-present as central.
    # G1 note: pattern-level CSR reclassification (the AASM v3 rule
    # "≥3 consecutive central + crescendo-decrescendo + ≥40 s cycle")
    # is detected by ancillary.detect_cheyne_stokes() and applied
    # downstream by postprocess.reclassify_csr_events() — not here.
    # Rule 5b is per-event ECG-based effort reclassification only.
    if ecg_assessment is not None and ecg_assessment.get("reclassify_as_central"):
        detail["ecg_assessment"] = {
            k: v for k, v in ecg_assessment.items()
            if k not in ("tecg_detail", "spectral_detail")
        }
        # v0.8.30: relaxed threshold from 1.5× to 2× EFFORT_PRESENT
        if effort_ratio < EFFORT_PRESENT_RATIO * 2.0:
            conf = 0.75
            if ecg_assessment.get("ecg_effort_present") is False:
                conf = 0.85  # both TECG and spectral agree
            detail["decision_reason"] = (
                f"ecg_reclassified_central_effort={safe_r(effort_ratio,3)}"
            )
            return "central", safe_r(_conf(conf, 5), 2), detail

    # ── Rule 6: Borderline default ────────────────────────────────────────
    # v0.8.30: if effort is in the low-but-not-absent range and no
    # clear obstructive evidence, classify as central rather than
    # defaulting to obstructive. This catches cardiac-pulsation-only
    # events that Rule 5a missed (typical effort_ratio 0.10-0.40 with
    # low raw_var). The v0.4.4 review noted that this is a deliberate
    # deviation from the AASM "when in doubt, obstructive" convention,
    # documented here so it is inspectable. To revert to AASM-strict
    # behaviour, lower the 0.40 threshold below to 0.30.
    if (
        effort_ratio < EFFORT_PRESENT_RATIO and  # 0.40 (AASM-deviation, see above)
        raw_var_ratio < 0.30 and
        not is_paradox and
        no_phase_signal
    ):
        conf_6 = 0.35 + _flat_central_boost
        detail["decision_reason"] = (
            f"low_effort_default_central_var={safe_r(raw_var_ratio,3)}_effort={safe_r(effort_ratio,3)}"
        )
        return "central", safe_r(_conf(conf_6, 6), 2), detail

    # Final default: obstructive (AASM "when in doubt, obstructive" convention).
    # Flattening boost lifts confidence when flow limitation is present.
    conf_6 = 0.40 + _flat_obstr_boost
    detail["decision_reason"] = (
        f"borderline_default_var={safe_r(raw_var_ratio,3)}_effort={safe_r(effort_ratio,3)}"
    )
    return "obstructive", safe_r(_conf(conf_6, 6), 2), detail


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_phase_angle(
    thorax_raw:  np.ndarray | None,
    abdomen_raw: np.ndarray | None,
    onset_idx:   int,
    end_idx:     int,
    sf:          float,
    min_dur_s:   float = 5.0,
) -> float | None:
    """
    Bereken de gemiddelde instantane fasehoek (in graden) tussen thorax en
    abdomen via de Hilbert-transformatie.

    0°   = perfect synchroon (normaal)
    90°  = kwartslag fase-verschuiving
    180° = volledig paradoxaal

    Een waarde >= 45° bij een flow-limitatie is een betrouwbare indicator
    van obstructief effort, ook wanneer de amplitude-envelop laag is.

    Vereist minimaal min_dur_s seconden signaal voor betrouwbare Hilbert.
    """
    if thorax_raw is None or abdomen_raw is None:
        return None
    seg_len = end_idx - onset_idx
    if seg_len < int(sf * min_dur_s):
        return None

    t_seg = thorax_raw[onset_idx:end_idx].astype(float)
    a_seg = abdomen_raw[onset_idx:end_idx].astype(float)

    # Verwijder DC-offset
    t_seg = t_seg - np.mean(t_seg)
    a_seg = a_seg - np.mean(a_seg)

    if np.std(t_seg) < 1e-9 or np.std(a_seg) < 1e-9:
        return None

    try:
        # Instantane fase via Hilbert-transformatie
        phi_t = np.angle(hilbert(t_seg))
        phi_a = np.angle(hilbert(a_seg))

        # Fase-verschil (gewikkeld naar [-π, π])
        delta_phi = np.angle(np.exp(1j * (phi_t - phi_a)))

        # Gemiddelde absolute fasehoek in graden
        mean_angle_deg = float(np.degrees(np.mean(np.abs(delta_phi))))
        return safe_r(mean_angle_deg, 1)
    except Exception:
        return None


def _compute_raw_variability(
    thorax_raw: np.ndarray | None,
    abdomen_raw: np.ndarray | None,
    onset_idx: int,
    end_idx: int,
    sf: float,
) -> float:
    """Bereken ruwe signaalvariabiliteit (standaarddeviatie) van effort-kanaal."""
    if thorax_raw is None and abdomen_raw is None:
        return 0.0
    event_stds = []
    for raw in (thorax_raw, abdomen_raw):
        if raw is not None:
            event_stds.append(float(np.std(raw[onset_idx:end_idx])))
    raw_variability = float(np.mean(event_stds))
    pre_start = max(0, onset_idx - int(120 * sf))
    pre_end   = max(0, onset_idx - int(5   * sf))
    if pre_end <= pre_start:
        return 1.0
    bl_stds = []
    for raw in (thorax_raw, abdomen_raw):
        if raw is not None:
            bl_stds.append(float(np.std(raw[pre_start:pre_end])))
    bl_var = max(float(np.mean(bl_stds)), 1e-9) if bl_stds else 1e-9
    return raw_variability / bl_var


def _compute_paradox_correlation(
    thorax_raw: np.ndarray | None,
    abdomen_raw: np.ndarray | None,
    onset_idx: int,
    end_idx: int,
) -> float | None:
    """Bereken paradoxale ademhalingscorrelatie tussen thorax en abdomen."""
    if thorax_raw is None or abdomen_raw is None:
        return None
    t_seg = thorax_raw[onset_idx:end_idx]
    a_seg = abdomen_raw[onset_idx:end_idx]
    if len(t_seg) <= 10 or np.std(t_seg) < 1e-9 or np.std(a_seg) < 1e-9:
        return None
    try:
        corr, _ = pearsonr(t_seg, a_seg)
        return float(corr)
    except Exception:
        return None


def _mean_effort_ratio(
    effort_segs: dict[str, np.ndarray],
    start: int,
    end: int,
    effort_baseline: float,
) -> float:
    """Gemiddelde effort-ratio: event-amplitude / basislijn-amplitude."""
    if effort_baseline < 1e-9:
        return 0.0
    vals = [float(np.mean(seg[start:end])) for seg in effort_segs.values()]
    return float(np.mean(vals)) / effort_baseline if vals else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Hypopneu-subtypering: de eigen regel van de manual
# ═══════════════════════════════════════════════════════════════════════════

#: Hoeveel de inspiratoire afvlakking tijdens het event moet TOENEMEN ten
#: opzichte van de basislijnademteugen om als obstructiekenmerk te tellen.
#: De manual zegt "toegenomen" en kwantificeert niet; dit is dus een keuze en
#: geen regel. Relatief, niet absoluut: een patiënt die de hele nacht op 0,35
#: ademt heeft geen toename op het event, en een absolute drempel zou daar
#: elke hypopneu obstructief noemen.
HYPOPNEA_FLATTENING_RATIO = 1.30

#: Fasehoek waarboven thorax en abdomen als paradoxaal gelden. Zelfde waarde
#: als in `classify_apnea_type` (regel 0), zodat er niet twee definities van
#: paradox naast elkaar bestaan.
HYPOPNEA_PARADOX_DEG = 45.0

#: Venster vóór het event waarin de "basislijnademhaling" wordt gemeten.
#: Gelijk aan het pre-event-basislijnvenster van AASM (2 minuten).
HYPOPNEA_BASELINE_S = 120.0


def classify_hypopnea_type(
    *,
    onset_s: float,
    duration_s: float,
    breaths: list | None,
    thorax_env: np.ndarray | None,
    abdomen_env: np.ndarray | None,
    sf: float,
    snore_present: bool | None = None,
    baseline_s: float = HYPOPNEA_BASELINE_S,
) -> tuple[str, float, dict]:
    """Subtypeer een hypopneu volgens AASM v3 §6.1 (optionele criteria).

    WAAROM DIT NIET `classify_apnea_type` IS
    ----------------------------------------
    Die functie implementeert sectie 3B, de APNEUregel, en beslist op
    effort-vlakheid: "no raw movement, low envelope" -> centraal. Bij een
    hypopneu is de inspanning per definitie nooit vlak -- de flow daalt 30 tot
    90 %, niet naar nul. Die regel kan daar dus vrijwel alleen vuren wanneer het
    EFFORTKANAAL zwak is, en dan zegt `hypopnea_central` iets over de
    meetopstelling in plaats van over de patiënt.

    De regels delen geen logica; parametriseren zou de fout verplaatsen in
    plaats van hem weghalen.

    DE REGEL, OMGEKEERD ONTWORPEN
    -----------------------------
    **Obstructief** bij ten minste één van:

    1. snurken tijdens het event;
    2. toegenomen inspiratoire afvlakking t.o.v. de basislijnademteugen;
    3. thoracoabdominale paradox die TIJDENS het event optreedt maar NIET in de
       ademhaling ervóór.

    **Centraal** alleen als geen van de drie aanwezig is. Eén kenmerk is
    genoeg -- geen stemming, geen weging.

    Criterium 3 draagt de bescherming: een band die de hele nacht al paradoxaal
    staat (verwisselde polariteit, losgekoppeld) is vóór het event óók
    paradoxaal en telt dus niet mee. Een absolute paradoxtoets zou daar juist
    wél vuren.

    WAT ONBEREIKBAAR IS EN WAAROM DAT IN DE UITVOER STAAT
    -----------------------------------------------------
    Criterium 1 vraagt snurken. Het bandfilter van 0,05-3 Hz knipt
    snurktrillingen er juist uit, dus zonder apart snurkkanaal is dit criterium
    op dit pad niet te toetsen. `snore_present=None` betekent dan ook niet
    "geen snurken" maar "niet gemeten", en het verschil staat in
    `criteria_unavailable`. Met twee van de drie criteria betekent "centraal"
    strikt genomen "geen van de twee die we kónden toetsen", en dat is zwakker
    dan de manual bedoelt. `complete` zegt of het oordeel op alle drie rust.

    Returns
    -------
    ``(subtype, confidence, detail)`` met subtype ``"obstructive"``,
    ``"central"`` of ``"uncertain"``.
    """
    detail: dict = {"rule": "AASM v3 §6.1 (optional)",
                    "criteria_met": [], "criteria_unavailable": []}

    # ── 1. Snurken ────────────────────────────────────────────────────────
    if snore_present is None:
        detail["criteria_unavailable"].append("snoring")
    elif snore_present:
        detail["criteria_met"].append("snoring")

    # ── 2. Toegenomen inspiratoire afvlakking ─────────────────────────────
    einde_s = onset_s + duration_s
    tijdens = [b["flattening"] for b in (breaths or [])
               if b.get("flattening") is not None
               and onset_s <= float(b.get("onset_s", -1)) < einde_s]
    basis = [b["flattening"] for b in (breaths or [])
             if b.get("flattening") is not None
             and onset_s - baseline_s <= float(b.get("onset_s", -1)) < onset_s]
    if tijdens and basis:
        f_ev = float(np.mean(tijdens))
        f_bl = float(np.mean(basis))
        detail["flattening_event"] = safe_r(f_ev, 3)
        detail["flattening_baseline"] = safe_r(f_bl, 3)
        if f_bl > 1e-9 and f_ev >= HYPOPNEA_FLATTENING_RATIO * f_bl:
            detail["criteria_met"].append("flattening")
    else:
        detail["criteria_unavailable"].append("flattening")

    # ── 3. Paradox tijdens, maar niet ervóór ──────────────────────────────
    if thorax_env is not None and abdomen_env is not None and sf > 0:
        o_i, e_i = int(onset_s * sf), int(einde_s * sf)
        p_i = max(0, int((onset_s - baseline_s) * sf))
        hoek_ev = _compute_phase_angle(thorax_env, abdomen_env, o_i, e_i, sf)
        hoek_bl = _compute_phase_angle(thorax_env, abdomen_env, p_i, o_i, sf)
        detail["phase_angle_event"] = safe_r(hoek_ev, 1)
        detail["phase_angle_baseline"] = safe_r(hoek_bl, 1)
        if hoek_ev is None or hoek_bl is None:
            detail["criteria_unavailable"].append("paradox")
        elif hoek_ev >= HYPOPNEA_PARADOX_DEG and hoek_bl < HYPOPNEA_PARADOX_DEG:
            detail["criteria_met"].append("paradox")
    else:
        detail["criteria_unavailable"].append("paradox")

    detail["complete"] = not detail["criteria_unavailable"]

    # ── Het oordeel ───────────────────────────────────────────────────────
    if len(detail["criteria_unavailable"]) == 3:
        # Niets te toetsen. `uncertain` in plaats van een restcategorie die
        # dan alleen zegt dat er niets gemeten is.
        return "uncertain", 0.3, detail
    if detail["criteria_met"]:
        # Meer kenmerken maken het niet obstructiever -- de regel is "ten
        # minste één" -- maar ze maken de uitspraak wel steviger.
        conf = 0.70 + 0.10 * min(2, len(detail["criteria_met"]) - 1)
        return "obstructive", round(conf, 2), detail

    # GEEN RESTCATEGORIE ZONDER ALLE DRIE DE CRITERIA.
    #
    # "Centraal" is bij deze regel wat overblijft als geen obstructiekenmerk
    # vuurt. Dat klopt alleen wanneer alle drie de kenmerken ook getoetst zijn.
    # Kon er één niet worden getoetst, dan betekent "geen kenmerk gevonden"
    # niet meer dan "we hebben er twee gekeken", en dat is geen centrale
    # hypopneu maar een onbekende.
    #
    # Gemeten op PSG-IPA (5 opnames, 2026-09-01): met snurken onbereikbaar
    # kwam 70,2 % van de hypopneus als centraal uit deze regel, tegen 5,9 %
    # bij menselijke scoorders. De restbak liep vol met wat niet gemeten kon
    # worden.
    #
    # Snurken is op het gefilterde flowpad niet af te leiden: de energie in
    # 30-100 Hz uit de neusdruk scheidt hypopneus niet van flow-gematchte
    # normale ademhaling (AUC 0,596 / 0,484 / 0,314 op SN1/SN3/SN5 -- toeval).
    # Er is een echt snurkkanaal voor nodig.
    if not detail["complete"]:
        return "uncertain", 0.40, detail
    return "central", 0.65, detail
