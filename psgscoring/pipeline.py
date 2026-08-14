"""
psgscoring.pipeline
===================
MNE-facing master function that orchestrates the full pneumological analysis
on a single EDF recording.

This is the only module that imports MNE.  All signal-processing modules
below it work exclusively with NumPy arrays and sample rates.

Call graph
----------
run_pneumo_analysis
  ├─ utils.channel_map_from_user
  ├─ respiratory.detect_respiratory_events
  ├─ spo2.analyze_spo2
  ├─ ancillary.analyze_position / heart_rate / snore / detect_cheyne_stokes
  ├─ plm.analyze_plm
  ├─ [arousal_analysis.run_arousal_respiratory_analysis]   <- optional
  └─ respiratory.reinstate_rule1a_arousal_hypopneas
"""

from __future__ import annotations
import logging
import os

import numpy as np

try:
    import mne
    _MNE_AVAILABLE = True
except ImportError:
    _MNE_AVAILABLE = False

from .utils import channel_map_from_user
from .signal import (
    compute_dynamic_baseline, preprocess_flow,
)
from .indices import per_hour
from .respiratory import (
    detect_respiratory_events, reinstate_rule1a_arousal_hypopneas, _compute_summary,
)
from .spo2 import analyze_spo2, compute_hypoxic_burden
from .ancillary import (
    analyze_position, analyze_heart_rate, analyze_snore, detect_cheyne_stokes,
)
from .plm import analyze_plm
from .ventilation import compute_ventilatory_burden

# Arousal & RERA detection — in-package as of v0.8.0 (ported from YASAFlaskified).
# Kept behind a guard so a broken optional dep (e.g. lightgbm) never aborts scoring.
try:
    from .arousal import run_arousal_respiratory_analysis
    _AROUSAL_AVAILABLE = True
except Exception:  # noqa: BLE001
    _AROUSAL_AVAILABLE = False

logger = logging.getLogger("psgscoring.pipeline")


def _breath_env_overrides() -> dict:
    """Env-overrides voor de drie experimentele scoringswijzigingen.

    Ze staan default UIT omdat aanzetten het op PSG-IPA geijkte werkpunt
    verlaat. Deze haak bestaat om dat te kunnen METEN zonder profielen te
    muteren; hetzelfde patroon als PSGSCORING_BASELINE_MODE.

      PSGSCORING_BREATH_CAND_MIN_DUR   sec, laat kortere runs de gradering halen
      PSGSCORING_BREATH_AROUSAL_LATENCY  1 = arousal graderen op latentie
      PSGSCORING_BREATH_TEMPLATE_DUR     1 = sjabloon gebruikt ook de duur
    """
    out: dict = {}
    v = os.environ.get("PSGSCORING_BREATH_CAND_MIN_DUR")
    if v:
        try:
            out["candidate_min_duration_s"] = float(v)
        except ValueError:
            logger.warning("[pneumo 7b] ongeldige CAND_MIN_DUR %r, genegeerd", v)
    if os.environ.get("PSGSCORING_BREATH_AROUSAL_LATENCY") == "1":
        out["arousal_latency_grading"] = True
    if os.environ.get("PSGSCORING_BREATH_TEMPLATE_DUR") == "1":
        out["template_use_duration"] = True
    if out:
        logger.info("[pneumo 7b] experimentele overrides actief: %s", out)
    return out


def _normalise_arousal_block(block: dict) -> dict:
    """Zorg dat ``output["arousal"]["events"]`` altijd bestaat.

    Twee producenten met een verschillende vorm: de externe tak zet de events
    plat op ``["events"]``; run_arousal_respiratory_analysis() geeft ze genest
    terug in ``["arousals"]["events"]`` en zet geen platte sleutel. Alle
    consumenten lezen de PLATTE sleutel -- de Rule 1A-reinstatement hieronder,
    en in YASAFlaskified de EDF+-export en de event-API. In het
    auto-detectiepad kregen die dus stil een lege lijst (issue #16).
    """
    if not isinstance(block, dict):
        return block
    if not block.get("events"):
        nested = block.get("arousals")
        block["events"] = (list(nested.get("events") or [])
                           if isinstance(nested, dict) else [])
    return block


def _run_step(step_name: str, fn):
    """Run an ancillary analysis, degrading to a ``{success: False, error}`` dict
    on any exception instead of crashing the whole run.

    Mirrors the ``try/except`` already used by the AHI-interval, ML, CSR,
    hypoxic-burden and post-processing steps: one malformed/unexpected channel
    (e.g. a NaN-laden leg-EMG) must not abort the entire respiratory analysis.
    On success the return value is exactly ``fn()`` (output-preserving).
    """
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — deliberate graceful degradation
        logger.warning("[pneumo] step '%s' failed, continuing: %s", step_name, e)
        return {"success": False, "error": f"{step_name}_failed: {e}", "summary": {}}


def run_pneumo_analysis(
    raw,                                 # mne.io.BaseRaw
    hypno: list,
    channel_map: dict | None = None,
    artifact_epochs: list | None = None,
    scoring_profile: str = "aasm_v3_rec",
    arousal_events: list | None = None,
) -> dict:
    """
    Run the full pneumological analysis on a single PSG recording.

    Parameters
    ----------
    raw             : MNE Raw object (EDF already loaded)
    hypno           : string hypnogram list ['W','N1','N2','N3','R',...]
    channel_map     : optional manual channel overrides (UI selection)
    artifact_epochs : list of epoch indices with artefacts (from YASA)
    scoring_profile : Profile name. New canonical names (v0.4.0+):
                        - 'aasm_v3_rec'       (default, AASM 2023 Rule 1A)
                        - 'aasm_v3_strict', 'aasm_v3_sensitive'
                        - 'aasm_v2_rec', 'aasm_v1_rec'
                        - 'cms_medicare', 'mesa_shhs', 'chicago_1999'
                      Legacy names (deprecated, still work):
                        - 'strict', 'standard', 'sensitive'
    arousal_events  : optional list of pre-scored arousals to use for
                      Rule 1B reinstatement, bypassing internal
                      EEG-based detection. Each item must be a dict
                      with at least ``onset_s`` and ``duration_s``
                      keys (seconds). Use this for dataset-faithful
                      validation against scorer-derived AHI variants
                      that include arousal-coupled hypopneas
                      (e.g. NSRR ``ahi_hp3u``).

    Returns
    -------
    Nested dict with keys: meta, channel_availability, respiratory, spo2,
    position, heart_rate, snore, plm, arousal, cheyne_stokes.
    """
    # v0.4.0: resolve legacy aliases (strict/standard/sensitive) with
    # DeprecationWarning. New names pass through unchanged.
    from .profiles import resolve_profile_name
    resolved_name = resolve_profile_name(scoring_profile)

    from .constants import SCORING_PROFILES
    profile = SCORING_PROFILES.get(
        resolved_name,
        SCORING_PROFILES["aasm_v3_rec"],
    )
    logger.info(
        "[pneumo] Scoring profile: %s (%s)",
        resolved_name,
        profile["label"],
    )

    ch = channel_map_from_user(channel_map, raw.ch_names)

    output: dict = {
        "meta": {
            "channels_used": ch,
            "all_channels":  raw.ch_names,
            "sfreq":         raw.info["sfreq"],
            "duration_min":  round(raw.times[-1] / 60, 1),
            "scoring_profile": resolved_name,
            "scoring_label":   profile["label"],
            # Env-overrides die profielwaarden overrulen. Zonder dit betekent
            # dezelfde profielnaam op twee machines iets anders, en het
            # herkomstblok — dat juist bestaat om de UITVOERING te tonen —
            # zwijgt erover. Leeg is het normale geval.
            "env_overrides":   _breath_env_overrides(),
            "scoring_input_name": scoring_profile,  # ← what user actually passed (may be legacy)
            # v0.11.0 (A5): explicit hypopnea scoring criterion for the report
            # (AASM v3 VIII.D Note 1 — the criterion used must be stated).
            "hypopnea_criterion": _hypopnea_criterion_str(profile),
            "patient_info":  _parse_edf_patient_info(raw),
        },
        "channel_availability": {k: (v in raw.ch_names) for k, v in ch.items()},
    }

    def get(ch_type):
        """Haal een kanaal op uit de MNE raw data op basis van de kanaalmap."""
        name = ch.get(ch_type)
        if name and name in raw.ch_names:
            return raw.get_data(picks=[name])[0], raw.info["sfreq"]
        return None, None

    # ── Flow channels ──────────────────────────────────────────────────────
    flow_data,          sf_flow  = get("flow")
    flow_pressure_data, sf_fp    = get("flow_pressure")
    flow_therm_data,    sf_ft    = get("flow_thermistor")

    apnea_flow, hypop_flow, sf_apnea, sf_hypop = _resolve_flow_channels(
        flow_data, sf_flow,
        flow_pressure_data, sf_fp,
        flow_therm_data, sf_ft,
        ch, output,
        # Bij additief gebruik hoeft de kwaliteitstoets niet te blokkeren;
        # zie de toelichting bij _resolve_flow_channels.
        additive_thermistor=profile.get("DUAL_SENSOR_APNEA", False),
        thermistor_gate=profile.get("THERMISTOR_GATE", "envelope_agreement"),
    )

    # Dual-sensor gevraagd maar niet uitvoerbaar: terugvallen op een sensor en
    # dat vastleggen, zodat het rapport het kan melden in plaats van stil een
    # ander algoritme te draaien dan de gebruiker koos.
    if profile.get("DUAL_SENSOR_APNEA", False) and (
            flow_therm_data is None or flow_pressure_data is None):
        _only = (ch.get("flow_pressure") or ch.get("flow_thermistor")
                 or ch.get("flow") or "—")
        output["meta"]["dual_sensor_fallback"] = {
            "requested": True, "performed": False, "channel": _only,
            "reason": "single_flow_channel",
        }
        logger.warning("[pneumo] dual-sensor gevraagd maar er is maar een "
                       "flowkanaal (%s); teruggevallen op een sensor", _only)
    elif profile.get("DUAL_SENSOR_APNEA", False):
        output["meta"]["dual_sensor_fallback"] = {
            "requested": True, "performed": True, "channel": None,
            "reason": None,
        }

    # ── Flow-referentiekanaal (v0.14.1) ────────────────────────────────────
    # Apneudetectie is niet de enige afnemer van een flowsignaal: de
    # AHI-intervalsweep, de anker-basislijn, de arousal/RERA-koppeling, de
    # Cheyne-Stokes-detectie en de ventilatory burden lezen er ook een. Die
    # lazen allemaal het apneukanaal. Onder een additief profiel is dat de
    # thermistor — bewust ook wanneer die de kwaliteitstoets niet haalt, want
    # de tweede detectiepas maakt hem ongevaarlijk vóór de apneutelling. Die
    # vijf kennen geen tweede pas en erfden zo precies het vervangende falen
    # dat de additieve opzet vermijdt.
    #
    # Default "apnea" = ongewijzigd gedrag; golden blijft byte-identiek.
    _flow_ref = str(profile.get("FLOW_REFERENCE", "apnea") or "apnea").lower()
    if _flow_ref == "hypopnea":
        ref_flow, sf_ref = hypop_flow, sf_hypop
    else:
        ref_flow, sf_ref = apnea_flow, sf_apnea
    if ref_flow is None:                       # montage zonder bruikbaar kanaal
        ref_flow, sf_ref = apnea_flow, sf_apnea
    output["meta"]["flow_channels"]["reference_sensor"] = (
        output["meta"]["flow_channels"].get("hypopnea_sensor")
        if ref_flow is hypop_flow and _flow_ref == "hypopnea"
        else output["meta"]["flow_channels"].get("apnea_sensor"))

    # ── Other channels ─────────────────────────────────────────────────────
    thorax_data,  _        = get("thorax")
    abdomen_data, _        = get("abdomen")
    spo2_data,    sf_spo2  = get("spo2")
    pulse_data,   sf_pulse = get("pulse")
    pos_data,     sf_pos   = get("position")
    snore_data,   sf_snore = get("snore")
    leg_l_data,   sf_leg   = get("leg_l")
    leg_r_data,   _        = get("leg_r")

    # v0.5.1: optional RIPsum fallback for hypopnea channel when nasal
    # pressure is degraded (mesa_shhs profile). Clinical profiles default
    # to FLOW_FALLBACK_STRATEGY="none" and this is a no-op.
    sf_rip = raw.info["sfreq"]
    hypop_flow, sf_hypop = _maybe_apply_ripsum_fallback(
        hypop_flow, sf_hypop,
        thorax_data, abdomen_data, sf_rip,
        profile, output,
    )

    eeg_data, sf_eeg = _pick_eeg(raw, ch)
    emg_data         = _pick_emg(raw, ch)

    # v0.3.001 BUG2 MARKER: RIP pair quality gate (moved BEFORE event detection)
    # Detects failed thorax or abdomen RIP sensors before classification.
    # Prevents false obstructive-default when one sensor fails
    # (cf. Loos case, AZORG 2026).
    logger.info("[pneumo 1b+] RIP pair quality assessment...")
    try:
        from .signal_quality import compare_rip_pair
        if thorax_data is not None and abdomen_data is not None:
            sf_rip = raw.info.get("sfreq", 256) if hasattr(raw, "info") else 256
            output["signal_quality"] = compare_rip_pair(
                thorax_data, abdomen_data, sf_rip,
                scale_free=bool(profile.get("RIP_QUALITY_SCALE_FREE", False)),
            )
            sq_mode = output["signal_quality"].get("recommended_mode")
            sq_ratio = output["signal_quality"].get("energy_ratio", 1.0)
            if sq_mode != "bilateral":
                logger.warning(
                    "[RIP quality] mode=%s, ratio=%.1fx, working=%s",
                    sq_mode, sq_ratio,
                    output["signal_quality"].get("working_channel"),
                )
                for w in output["signal_quality"].get("warnings", []):
                    logger.warning("[RIP quality] %s", w)
        else:
            logger.info("[RIP quality] thorax or abdomen not available - skip")
            output["signal_quality"] = None
    except Exception as e:
        logger.warning("RIP pair quality check failed: %s", e)
        output["signal_quality"] = {"error": str(e)}

    # ── Step 1: Respiratory events ─────────────────────────────────────────
    logger.info("[pneumo 1/9] Apnea / hypopnea detection (AASM)...")
    # Extract ECG for effort-based apnea type classification (v0.8.23)
    ecg_data_resp, sf_ecg_resp = get("ecg")
    # perf (v0.7.x): shared-preprocessing cache reused by the primary call and
    # the 3-profile AHI-interval reruns below — envelopes / dynamic baseline /
    # position changes are computed once (byte-identical, same signals + params).
    _precomp: dict = {}
    if apnea_flow is not None:
        resp = detect_respiratory_events(
            flow_data    = apnea_flow,
            hypop_flow   = hypop_flow,
            sf_hypop     = sf_hypop,
            thorax_data  = thorax_data,
            abdomen_data = abdomen_data,
            spo2_data    = spo2_data,
            sf_flow      = sf_apnea,
            sf_spo2      = sf_spo2 or sf_flow or 1.0,
            hypno        = hypno,
            artifact_epochs = artifact_epochs,
            pos_data     = pos_data,
            sf_pos       = sf_pos,
            scoring_profile = profile,
            ecg_data     = ecg_data_resp,
            sf_ecg       = sf_ecg_resp,
            signal_quality = output.get("signal_quality"),  # v0.3.001 BUG2
            _precomputed = _precomp,
        )
    else:
        resp = {
            "success": False,
            "error":   "No flow channel found",
            "events":  [], "summary": {},
        }
    # ── Tweede sensorpas: apneus op het ANDERE flowkanaal ────────────────
    # Kiezen tussen thermistor en neusdruk is een valse keuze. De neusdruk
    # mist mondademhaling, de thermistor is te traag voor korte events. Dit
    # profiel detecteert apneus op beide en ontdubbelt op overlap, zodat
    # dezelfde gebeurtenis niet twee keer in de AHI belandt.
    #
    # Falsifieren gebeurt NIET tenzij DUAL_SENSOR_CORROBORATION aan staat:
    # een event dat maar een sensor ziet blijft staan. Bij twijfel wint
    # goedkeuren, want een gemiste centrale apneu weegt zwaarder.
    #
    # Hypopneeen komen ongewijzigd uit de eerste pas: die horen per AASM op
    # de neusdruk en hebben sinds de basislijnfix hun eigen basislijn.
    _second = None
    if (profile.get("DUAL_SENSOR_APNEA", False)
            and resp.get("success")
            and flow_therm_data is not None and flow_pressure_data is not None):
        _alt = (flow_pressure_data if apnea_flow is flow_therm_data
                else flow_therm_data)
        _alt_sf = (sf_fp if apnea_flow is flow_therm_data else sf_ft) or sf_apnea
        try:
            logger.info("[pneumo] tweede sensorpas voor apneus...")
            _second = detect_respiratory_events(
                flow_data=_alt, hypop_flow=hypop_flow, sf_hypop=sf_hypop,
                thorax_data=thorax_data, abdomen_data=abdomen_data,
                spo2_data=spo2_data, sf_flow=_alt_sf,
                sf_spo2=sf_spo2 or sf_flow or 1.0, hypno=hypno,
                artifact_epochs=artifact_epochs, pos_data=pos_data,
                sf_pos=sf_pos, scoring_profile=profile,
                ecg_data=ecg_data_resp, sf_ecg=sf_ecg_resp,
                signal_quality=output.get("signal_quality"),
            )
        except Exception as e:  # noqa: BLE001 — nooit de hele run laten vallen
            logger.warning("[pneumo] tweede sensorpas mislukt: %s", e)
            _second = None

    if _second is not None and _second.get("success"):
        from .postprocess import corroborate_apnea_events

        def _is_apnea(e):
            t = str(e.get("type", ""))
            return "hypopnea" not in t and (
                t in ("obstructive", "central", "mixed", "uncertain", "apnea")
                or "apnea" in t)

        _prim_ap = [e for e in resp.get("events", []) if _is_apnea(e)]
        _sec_ap = [e for e in _second.get("events", []) if _is_apnea(e)]
        _rest = [e for e in resp.get("events", []) if not _is_apnea(e)]
        _therm_first = apnea_flow is flow_therm_data
        _apneas, _cd = corroborate_apnea_events(
            _prim_ap if _therm_first else _sec_ap,
            _sec_ap if _therm_first else _prim_ap,
            corroboration_licensed=profile.get(
                "DUAL_SENSOR_CORROBORATION", False),
        )
        # v0.14.4 (§3.1): de overeenstemming als GEWICHT in plaats van poort.
        # assess_flow_sensor_agreement levert een continue envelope-correlatie
        # en die werd gereduceerd tot usable ja/nee — dezelfde vorm als de
        # arousal-as vóór deze versie: een drempel in een continu model. Hier
        # blijft het getal staan op elk event, en een apneu waarvan de
        # thermistor de enige steun is draagt de twijfel over die sensor mee
        # in zijn confidence in plaats van hem te verzwijgen.
        if profile.get("THERMISTOR_AGREEMENT_WEIGHTING", False):
            _agr = ((output.get("meta", {}) or {}).get("flow_channels", {})
                    or {}).get("thermistor_check") or {}
            _w = _agr.get("agreement")
            if isinstance(_w, (int, float)):
                _w = float(max(0.0, min(1.0, _w)))
                _n_w = 0
                for _e in _apneas:
                    _e["sensor_agreement"] = round(_w, 3)
                    if _e.get("corroboration") == "thermistor_only":
                        _c = _e.get("confidence")
                        if isinstance(_c, (int, float)):
                            _e["confidence"] = round(float(_c) * _w, 3)
                            _e["confidence_weighted_by"] = "sensor_agreement"
                            _n_w += 1
                logger.info("[pneumo] sensorovereenstemming %.2f als gewicht: "
                            "%d thermistor-only apneus herwogen", _w, _n_w)

        merged = sorted(_apneas + _rest,
                        key=lambda e: float(e.get("onset_s") or 0.0))
        output["respiratory"] = resp
        resp["events"] = merged
        resp["summary"] = _compute_summary(merged, hypno, artifact_epochs)
        resp["dual_sensor_apnea"] = _cd
        logger.info("[pneumo] dual-sensor apneus: %d beide, %d alleen "
                    "thermistor, %d alleen druk -> %d behouden",
                    _cd["n_both"], _cd["n_thermistor_only"],
                    _cd["n_pressure_only"], _cd["n_kept"])

    output["respiratory"] = resp

    # `dual_sensor` betekent: apneus en hypopneeën zijn op VERSCHILLENDE
    # sensoren gescoord. respiratory.py zette de vlag op True zodra er
    # überhaupt een hypopnee-kanaal was, zonder te vergelijken — en omdat
    # _resolve_flow_channels() bij een ontbrekende thermistor terugvalt op
    # hetzelfde kanaal, stond hij praktisch altijd aan. De enige consument
    # is de rapportnoot "apneu op thermistor, hypopneu op nasale druk", die
    # daardoor een methodiek claimde die niet was toegepast.
    #
    # meta["flow_channels"]["dual_sensor"] is wél correct berekend (beide
    # kanalen aanwezig); die is hier de bron van waarheid.
    _fc = (output.get("meta", {}) or {}).get("flow_channels", {}) or {}
    if isinstance(resp, dict) and "dual_sensor" in _fc:
        resp["dual_sensor"] = bool(_fc["dual_sensor"])

    # ── Step 1a (v0.2.8): AHI Confidence Interval ─────────────────────────
    # Run all 3 profiles and compute [strict–sensitive] interval + robustness
    logger.info("[pneumo 1a/11] AHI confidence interval (3 profiles)...")
    try:
        _all_profiles = ["strict", "standard", "sensitive"]
        _profile_results = {}
        # Reuse the primary result for whichever interval profile is the same as
        # the primary. Match on the resolved profile name (_PROFILE_NAME), not the
        # legacy alias string — fixes a bug where canonical names like
        # 'aasm_v3_rec' never equalled 'standard', so the primary was needlessly
        # re-scored as a 4th full pass.
        _primary_name = (profile or {}).get("_PROFILE_NAME")
        # De gedeelde voorbewerkingscache geldt alleen bij hetzelfde signaal.
        # Zodra de sweep een ánder kanaal leest dan de primaire pas, zouden de
        # envelope en de dynamische basislijn van het apneukanaal komen terwijl
        # de events op het referentiekanaal gescoord worden. Verse cache dan.
        _sweep_precomp = _precomp if ref_flow is apnea_flow else {}
        for _pname in _all_profiles:
            _alt_profile = SCORING_PROFILES.get(_pname, {})
            if _alt_profile.get("_PROFILE_NAME") == _primary_name:
                # Reuse primary result (identical profile)
                _profile_results[_pname] = resp.get("summary", {})
            else:
                # v0.14.1: het referentiekanaal, niet het apneukanaal. Dit is
                # een enkelsensor-herscoring; op een montage met een dode
                # thermistor stortte de strict-arm in en dat getal belandde als
                # "Rule 1B" in het rapport.
                _alt_resp = detect_respiratory_events(
                    flow_data    = ref_flow,
                    hypop_flow   = hypop_flow,
                    sf_hypop     = sf_hypop,
                    thorax_data  = thorax_data,
                    abdomen_data = abdomen_data,
                    spo2_data    = spo2_data,
                    sf_flow      = sf_ref,
                    sf_spo2      = sf_spo2 or sf_flow or 1.0,
                    hypno        = hypno,
                    artifact_epochs = artifact_epochs,
                    pos_data     = pos_data,
                    sf_pos       = sf_pos,
                    scoring_profile = _alt_profile,
                    ecg_data     = ecg_data_resp,
                    sf_ecg       = sf_ecg_resp,
                    signal_quality = output.get("signal_quality"),  # v0.3.001 BUG2
                    _precomputed = _sweep_precomp,
                )
                _profile_results[_pname] = _alt_resp.get("summary", {})

        def _severity(ahi):
            if ahi < 5: return "normal"
            if ahi < 15: return "mild"
            if ahi < 30: return "moderate"
            return "severe"

        _ahi_strict    = _profile_results.get("strict", {}).get("ahi_total", 0)
        _ahi_standard  = _profile_results.get("standard", {}).get("ahi_total", 0)
        _ahi_sensitive = _profile_results.get("sensitive", {}).get("ahi_total", 0)

        _sev_strict    = _severity(_ahi_strict)
        _sev_standard  = _severity(_ahi_standard)
        _sev_sensitive = _severity(_ahi_sensitive)

        _sevs = [_sev_strict, _sev_standard, _sev_sensitive]
        if _sevs[0] == _sevs[1] == _sevs[2]:
            _robustness = "A"
            _robustness_label = "Robust — all profiles concordant"
        elif _sevs.count(max(set(_sevs), key=_sevs.count)) >= 2:
            _robustness = "B"
            _robustness_label = "Probable — 2/3 profiles concordant"
        else:
            _robustness = "C"
            _robustness_label = "Uncertain — profiles discordant, manual review recommended"

        _oahi_strict    = _profile_results.get("strict", {}).get("oahi", _ahi_strict)
        _oahi_standard  = _profile_results.get("standard", {}).get("oahi", _ahi_standard)
        _oahi_sensitive = _profile_results.get("sensitive", {}).get("oahi", _ahi_sensitive)

        output["ahi_interval"] = {
            "strict":    {"ahi": round(_ahi_strict, 1),    "oahi": round(_oahi_strict, 1),    "severity": _sev_strict},
            "standard":  {"ahi": round(_ahi_standard, 1),  "oahi": round(_oahi_standard, 1),  "severity": _sev_standard},
            "sensitive": {"ahi": round(_ahi_sensitive, 1), "oahi": round(_oahi_sensitive, 1), "severity": _sev_sensitive},
            "interval":  [round(min(_ahi_strict, _ahi_standard, _ahi_sensitive), 1),
                          round(max(_ahi_strict, _ahi_standard, _ahi_sensitive), 1)],
            "robustness_grade": _robustness,
            "robustness_label": _robustness_label,
            "primary_profile":  scoring_profile,
        }
        # Backward compat: populate profile_comparison for PDF report
        output["respiratory"]["profile_comparison"] = {
            "strict":    {"oahi": round(_oahi_strict, 1)},
            "standard":  {"oahi": round(_oahi_standard, 1)},
            "sensitive": {"oahi": round(_oahi_sensitive, 1)},
        }
        logger.info("[pneumo] AHI interval: [%.1f – %.1f] (%s → %s → %s), robustness: %s",
                     _ahi_strict, _ahi_sensitive, _sev_strict, _sev_standard,
                     _sev_sensitive, _robustness)
    except Exception as e:
        logger.warning("AHI interval computation failed: %s", e)
        output["ahi_interval"] = {"error": str(e)}

    # ── Step 1b (v0.8.16): Signal quality assessment ─────────────────────
    logger.info("[pneumo 1b/11] Signal quality assessment...")
    try:
        from .signal_quality_channels import assess_signal_quality
        # v0.8.41: renamed output key signal_quality -> channel_quality
        # to avoid naming conflict with new RIP pair quality module.
        output["channel_quality"] = assess_signal_quality(raw, ch, hypno)
        cq = output["channel_quality"]
        if cq.get("montage_warnings"):
            for w in cq["montage_warnings"]:
                logger.warning("[quality] MONTAGE: %s", w)
        if cq.get("overall_grade") == "poor":
            logger.warning("[quality] Overall signal quality: POOR")
    except Exception as e:
        logger.warning("Channel quality assessment failed: %s", e)
        output["channel_quality"] = {"overall_grade": "unknown", "error": str(e)}



    # ── Step 1c (v0.8.11): Baseline Anchoring ─────────────────────────────
    if ref_flow is not None:
        try:
            from .signal import compute_anchor_baseline  # not in the top-level import
            _anchor_env = preprocess_flow(ref_flow, sf_ref, is_nasal_pressure=False)
            anchor_info = compute_anchor_baseline(
                _anchor_env, sf_ref, hypno,
                events=resp.get("events", []),
                artifact_epochs=artifact_epochs,
            )
            output["anchor_baseline"] = anchor_info
            if anchor_info.get("mouth_breathing_suspected"):
                logger.warning(
                    "Mond-ademen verdacht: anchor_ratio=%.2f — hypopnea-confidence mogelijk verlaagd",
                    anchor_info.get("anchor_ratio", 0),
                )
        except Exception as e:
            logger.debug("Anchor baseline failed: %s", e)
            output["anchor_baseline"] = {"anchor_reliable": False}
    else:
        output["anchor_baseline"] = {"anchor_reliable": False}

    # ── Step 2: SpO2 ───────────────────────────────────────────────────────
    logger.info("[pneumo 2/11] SpO2 analysis...")
    if spo2_data is not None:
        output["spo2"] = _run_step("spo2", lambda: analyze_spo2(spo2_data, sf_spo2, hypno))
        # v0.8.16: SpO2 samplerate check (AASM: max 3s averaging)
        if sf_spo2 is not None and sf_spo2 < 0.33:
            output["spo2"]["spo2_low_samplerate"] = True
            logger.warning("[pneumo] SpO2 samplerate %.2f Hz < 0.33 Hz "
                           "(>3s averaging — may underestimate ODI)", sf_spo2)
        output["spo2"]["spo2_samplerate"] = sf_spo2
    else:
        output["spo2"] = {"success": False, "error": "No SpO2 channel", "summary": {}}

    # ── Step 3: Position ───────────────────────────────────────────────────
    logger.info("[pneumo 3/9] Position analysis...")
    output["position"] = (
        _run_step("position", lambda: analyze_position(pos_data, sf_pos, hypno, resp.get("events", [])))
        if pos_data is not None
        else {"success": False, "error": "No position channel", "summary": {}}
    )

    # ── Step 4: Heart rate ─────────────────────────────────────────────────
    logger.info("[pneumo 4/9] Heart rate...")
    # Zonder pulse-kanaal valt dit terug op het ruwe ECG. Dat is geen bpm: de
    # golfvorm gaat ongewijzigd door het plausibiliteitsfilter en alles tussen
    # 20 en 250 in ADC-eenheden overleeft als "hartfrequentie". De rol gaat mee
    # naar analyze_heart_rate, dat de uitkomst dan als onbetrouwbaar markeert.
    # hr_data zelf blijft ongewijzigd — de arousal-analyse leest hetzelfde
    # signaal, en dat mag hier niet verschuiven.
    if pulse_data is not None:
        hr_data, sf_hr, _hr_source = pulse_data, sf_pulse, "pulse"
    else:
        hr_data, sf_hr = get("ecg")
        _hr_source = "ecg"
    output["heart_rate"] = (
        _run_step("heart_rate",
                  lambda: analyze_heart_rate(hr_data, sf_hr, hypno,
                                             source=_hr_source))
        if hr_data is not None
        else {"success": False, "error": "No HR/ECG channel", "summary": {}}
    )

    # ── Step 5: Snore ──────────────────────────────────────────────────────
    logger.info("[pneumo 5/9] Snore analysis...")
    output["snore"] = (
        _run_step("snore", lambda: analyze_snore(snore_data, sf_snore, hypno))
        if snore_data is not None
        else {"success": False, "error": "No snore channel", "summary": {}}
    )

    # ── Step 6: PLM ────────────────────────────────────────────────────────
    logger.info("[pneumo 6/9] PLM detection...")
    if leg_l_data is not None or leg_r_data is not None:
        output["plm"] = _run_step("plm", lambda: analyze_plm(
            leg_l_data, leg_r_data,
            sf_leg or raw.info["sfreq"], hypno,
            resp_events=resp.get("events", []),
            artifact_epochs=artifact_epochs,
        ))
    else:
        output["plm"] = {"success": False, "error": "No leg-EMG channels", "summary": {}}

    # ── Step 7: Arousal detection ──────────────────────────────────────────
    if arousal_events is not None:
        logger.info(
            "[pneumo 7/9] Using %d externally provided arousals "
            "(EEG-based detection skipped)",
            len(arousal_events),
        )
        ext = _empty_arousal("external_arousals")
        ext["success"] = True
        ext["error"]   = None
        ext["source"]  = "external"
        ext["events"]  = list(arousal_events)
        # Compute arousal index against TST from hypno (best-effort).
        from .constants import EPOCH_LEN_S as _EPOCH
        n_tst_eps = sum(1 for s in hypno if s in ("N1", "N2", "N3", "R"))
        tst_h     = (n_tst_eps * _EPOCH) / 3600.0 if n_tst_eps else 0.0
        ext["summary"]["arousal_index"] = (
            round(len(arousal_events) / tst_h, 2) if tst_h > 0 else None
        )
        output["arousal"] = ext
    elif eeg_data is not None and _AROUSAL_AVAILABLE:
        logger.info("[pneumo 7/9] Arousal detection & respiratory coupling...")
        flow_env_norm = _compute_flow_norm(ref_flow, sf_ref)
        # v0.9.0: arousal-afleidingsmodus. Clinical profielen defaulten naar 'multi'
        # (centraal + occipitaal + frontaal, event-level union + EOG-reject); dataset-
        # profielen (mesa_shhs) blijven 'single' voor NSRR-reproductie. Env
        # `PSGSCORING_AROUSAL_DERIVATION` overschrijft het profiel; bij < 2 bruikbare
        # afleidingen degradeert 'multi' vanzelf naar single (byte-identiek).
        _ar_mode = (os.environ.get("PSGSCORING_AROUSAL_DERIVATION")
                    or profile.get("AROUSAL_DERIVATION_MODE", "single")).lower()
        _derivations = None
        _eog_arousal = None
        _eog_reject = False
        if _ar_mode == "multi":
            _multi = _pick_eeg_multi(raw, ch)
            if len(_multi) > 1:
                _derivations = _multi
                _eog_arousal = _pick_eog(raw, ch)
                # Human-like: drop occipital-only events coinciding with an eye
                # movement (EOG-doorslag). On by default in multi mode when EOG is
                # available; disable with PSGSCORING_AROUSAL_EOG_REJECT=0.
                _eog_reject = (os.environ.get("PSGSCORING_AROUSAL_EOG_REJECT", "1") == "1"
                               and _eog_arousal is not None)
                logger.info("[pneumo] multi-derivatie arousal over %s (eog_reject=%s)",
                            [n for n, _d, _s in _multi], _eog_reject)
        try:
            output["arousal"] = run_arousal_respiratory_analysis(
                eeg_data    = eeg_data,
                sf_eeg      = sf_eeg,
                flow_data   = ref_flow,
                flow_norm   = flow_env_norm,
                sf_flow     = sf_ref,
                resp_events = resp.get("events", []),
                hypno       = hypno,
                emg_data    = emg_data,
                artifact_epochs = artifact_epochs,
                hr_data     = hr_data,
                sf_hr       = sf_hr or 1.0,
                derivations = _derivations,
                eog_data    = _eog_arousal,
                eog_reject  = _eog_reject,
            )
        except Exception as e:  # noqa: BLE001 — arousal failure must not abort the run
            logger.warning("[pneumo] arousal analysis failed, continuing: %s", e)
            output["arousal"] = _empty_arousal(f"arousal_failed: {e}")
    else:
        reason = "No EEG channel" if eeg_data is None else "arousal_analysis not loaded"
        output["arousal"] = _empty_arousal(reason)

    # Eén vorm voor alle consumenten, ongeacht welke producent hierboven liep.
    output["arousal"] = _normalise_arousal_block(output["arousal"])

    # ── Step 7b: ademteug-gebaseerde hypopnee-detector (opt-in) ───────────
    # Vervangt ALLEEN de hypopneeën; apneus blijven uit de bestaande detector
    # (die haalt F1 0,83-0,93 op PSG-IPA en doet de effort-subtypering).
    if str(profile.get("HYPOPNEA_DETECTOR", "envelope")) == "breath_graded":
        try:
            from .breath_scoring import score_hypopneas_breathwise
            _resp = output.get("respiratory", {}) or {}
            _all = _resp.get("events", []) or []
            _apneas = [e for e in _all
                       if "hypopnea" not in str(e.get("type", "")).lower()]
            _excl = [(float(e["onset_s"]),
                      float(e["onset_s"]) + float(e.get("duration_s") or 0))
                     for e in _apneas if e.get("onset_s") is not None]
            _ar = (output.get("arousal") or {}).get("events") or []
            _hyps, _bdiag = score_hypopneas_breathwise(
                breaths=_resp.get("_breaths", []),
                hypno=hypno,
                spo2=spo2_data,
                sf_spo2=sf_spo2 or 1.0,
                arousals=_ar,
                exclude_intervals=_excl,
                flow_reduction_threshold=1.0 - float(profile.get("HYPOPNEA_THRESHOLD", 0.70)),
                min_duration_s=float(profile.get("HYPOPNEA_MIN_DUR_S", 10.0)),
                max_duration_s=float(profile.get("HYPOPNEA_MAX_DUR_S", 60.0)),
                desat_threshold_pct=float(profile.get("DESATURATION_DROP_PCT", 3.0)),
                desat_low_baseline_relaxation=bool(
                    profile.get("DESAT_LOW_BASELINE_RELAXATION", False)),
                strictness=float(profile.get("HYPOPNEA_STRICTNESS", 0.50)),
                # v0.14.4: de arousal-tak is nu profielgestuurd. Hij was de
                # enige as die niet gegradeerd was — p_arousal sprong naar het
                # gewicht zodra er een arousal in het venster lag, waardoor
                # p_confirm nooit onder dat gewicht kwam ongeacht de
                # desaturatie. Zie hypopnea_arousal_weight.
                arousal_weight=float(profile.get("HYPOPNEA_AROUSAL_WEIGHT", 0.90)),
                desat_width=float(profile.get("HYPOPNEA_DESAT_WIDTH", 0.80)),
                arousal_latency_grading=bool(
                    profile.get("HYPOPNEA_AROUSAL_LATENCY", False)),
                # Het koppelvenster was op DEZE tak niet profielgestuurd: het
                # profielveld bereikte alleen de Rule 1B-herstelpas (regel ~900),
                # terwijl `score_hypopneas_breathwise` op zijn eigen default van
                # 15 s draaide. Dezelfde grootheid stond dus op twee plekken en
                # kon uiteenlopen zonder dat iets dat merkte.
                #
                # Doorbedraden is byte-identiek: het enige profiel met een
                # afwijkend venster is `mesa_shhs` (5 s), en dat draait de
                # envelope-detector, niet deze tak.
                arousal_window_s=float(
                    profile.get("RULE1B_AROUSAL_WINDOW_S", 15.0)),
                # Env-overrides voor de drie scoringswijzigingen, zodat ze
                # gemeten kunnen worden zonder profielen te muteren —
                # zelfde patroon als PSGSCORING_BASELINE_MODE. Ze staan NA de
                # profielwaarden en winnen dus, zodat bestaande metingen
                # blijven werken. Niet gezet = gedrag van v0.13.0.
                **_breath_env_overrides(),
            )
            # Subtypering overerven. De envelope-detector classificeerde
            # dezelfde tijdvensters al met classify_apnea_type() — effort,
            # ECG, flattening — en die events worden hier weggegooid. Dat
            # label opnieuw berekenen zou de enveloppes moeten reconstrueren
            # die alleen binnen respiratory.py bestaan; overerven kost niets
            # en behoudt hypopnea_central / hypopnea_mixed in het rapport.
            # Waar geen envelope-event overlapt blijft het "hypopnea".
            _env_sub = [
                (float(e["onset_s"]),
                 float(e["onset_s"]) + float(e.get("duration_s") or 0.0),
                 str(e.get("type", "")))
                for e in _all
                if "hypopnea" in str(e.get("type", "")).lower()
                and str(e.get("type", "")) != "hypopnea"
                and e.get("onset_s") is not None
            ]
            _n_sub = 0
            for _h in _hyps:
                _a = float(_h["onset_s"])
                _b = _a + float(_h["duration_s"])
                _best, _ov = None, 0.0
                for _sa, _sb, _lab in _env_sub:
                    _o = min(_b, _sb) - max(_a, _sa)
                    if _o > _ov:
                        _best, _ov = _lab, _o
                if _best is not None:
                    _h["type"] = _best
                    _h["subtype_source"] = "inherited_from_envelope_detector"
                    _n_sub += 1
            _bdiag["n_subtyped"] = _n_sub
            _bdiag["n_envelope_subtyped"] = len(_env_sub)

            merged = sorted(_apneas + _hyps, key=lambda e: float(e["onset_s"]))
            output["respiratory"]["events"] = merged
            output["respiratory"]["summary"] = _compute_summary(
                merged, hypno, artifact_epochs)
            output["respiratory"]["breath_detector"] = _bdiag
            logger.info("[pneumo 7b] breath-graded hypopneas: %d scored from "
                        "%d candidates (SpO2 lag %.0fs)",
                        _bdiag.get("n_scored", 0), _bdiag.get("n_candidates", 0),
                        _bdiag.get("spo2_lag_s") or -1)

            # De positieanalyse draaide in stap 6 op de eventlijst van DAT
            # moment. Deze stap heeft zojuist elke hypopnee vervangen, dus de
            # AHI-per-positie sloeg nog op de envelope-detector terwijl de
            # rest van het rapport de gegradeerde events toont. Dat raakt elk
            # profiel met breath_graded — breath, prob en beide duale — en
            # daarmee ook de positionele fenotypering verderop, die
            # `ahi_per_pos` leest.
            #
            # Herberekenen in plaats van corrigeren: de positieverdeling zelf
            # verandert niet, alleen de events erin, en analyze_position is de
            # enige plek waar die koppeling gelegd wordt.
            if pos_data is not None and sf_pos:
                try:
                    output["position"] = analyze_position(
                        pos_data, sf_pos, hypno, merged)
                    logger.info("[pneumo 7b] positieanalyse herrekend op de "
                                "gegradeerde eventlijst (%d events)", len(merged))
                except Exception as e:  # noqa: BLE001
                    logger.warning("[pneumo 7b] positieanalyse niet te "
                                   "herrekenen, oude waarden blijven: %s", e)
        except Exception as e:  # noqa: BLE001 — nooit de hele run laten vallen
            logger.warning("[pneumo] breath-graded detector failed, keeping "
                           "envelope result: %s", e)

    # ── Step 8: Rule 1B reinstatement ─────────────────────────────────────
    logger.info("[pneumo 8/10] Rule 1B reinstatement...")
    rejected  = resp.get("rejected_hypopneas", [])
    # Zie PostProcessingRules.arousal_limb_wired. De plumbing is gerepareerd
    # (issue #16), maar WELK profiel daarop reageert is een profielkeuze met
    # het bestaande gedrag als default: arousal-afhankelijke stappen -
    # Rule 1B-reinstatement, FRI-RERA en de LightGBM-features - zouden anders
    # bestaande klinische en dataset-uitkomsten veranderen zonder dat elk
    # onderdeel afzonderlijk opnieuw gevalideerd is.
    #
    # Step 7b hierboven leest output["arousal"]["events"] rechtstreeks en
    # heeft zijn arousal-tak dus wel, ongeacht deze vlag.
    _ar_block = output.get("arousal", {}) or {}
    if _ar_block.get("source") == "external":
        # Door de aanroeper meegegeven arousals zijn nooit door issue #16
        # geraakt en worden altijd gehonoreerd: wie ze expliciet meegeeft,
        # vraagt erom. Dit is het pad dat cms_arousal in de golden dekt.
        arousals = list(_ar_block.get("events") or [])
    elif profile.get("AROUSAL_LIMB_WIRED", False):
        arousals = list(_ar_block.get("events") or [])
    else:
        arousals = []
    # Rule 1B reinstates arousal-coupled hypopneas that lacked a qualifying
    # desaturation. Profiles that score hypopneas on desaturation ONLY (no
    # arousal qualifier) — cms_medicare, aasm_v1_rec (DESAT_OR_AROUSAL=False) —
    # must not do this, or their AHI is inflated by arousal-only events.
    allow_arousal = profile.get("DESAT_OR_AROUSAL", True)
    # Env-override zodat fase 4 de 2x2 kan meten zonder profielen te
    # muteren; zelfde patroon als PSGSCORING_BASELINE_MODE.
    _limb_env = os.environ.get("PSGSCORING_RULE1A_AROUSAL")
    limb_enabled  = (_limb_env.strip().lower() in ("1", "true", "yes", "on")
                     if _limb_env is not None
                     else bool(profile.get("RULE1A_AROUSAL_ENABLED", False)))

    # v0.12.3+: expliciete telling. De tak stond jarenlang op nul zonder dat
    # iets dat meldde; vanaf nu is 'nul' altijd zichtbaar mét reden.
    ar_stats = {
        "enabled":              limb_enabled,
        "n_arousals_available": len(arousals),
        "n_candidates_tested":  0,
        "n_arousal_coupled":    0,
        "n_qualified":          0,
        "skipped_reason":       None,
    }
    if not limb_enabled:
        ar_stats["skipped_reason"] = "disabled_by_profile"
    elif not allow_arousal:
        ar_stats["skipped_reason"] = "profile_scores_desaturation_only"
    elif not arousals:
        ar_stats["skipped_reason"] = "no_arousals_detected"
    elif not rejected:
        ar_stats["skipped_reason"] = "no_rejected_candidates"

    if rejected and arousals and allow_arousal and limb_enabled:
        reinstated, updated_events = reinstate_rule1a_arousal_hypopneas(
            rejected       = rejected,
            arousal_events = arousals,
            resp_events    = resp.get("events", []),
            hypno          = hypno,
            breaths        = resp.get("_breaths", []),
            arousal_window_s = profile.get("RULE1B_AROUSAL_WINDOW_S"),
            gap_max_breaths  = int(profile.get("RULE1A_GAP_MAX_BREATHS", 1)),
            stats            = ar_stats,
        )
        if reinstated:
            logger.info("[pneumo 8] Rule 1B: %d hypopneas reinstated", len(reinstated))
            output["respiratory"]["events"]           = updated_events
            output["respiratory"]["summary"]          = _compute_summary(
                updated_events, hypno, artifact_epochs
            )
            # v0.12.3+: 1A is de correcte AASM-regel; de 1B-sleutel blijft
            # als alias omdat YASAFlaskified hem in twee rapporten leest.
            output["respiratory"]["rule1a_arousal_reinstated"] = len(reinstated)
            output["respiratory"]["rule1b_reinstated"] = len(reinstated)
    else:
        output["respiratory"].setdefault("rule1a_arousal_reinstated", 0)
        output["respiratory"].setdefault("rule1b_reinstated", 0)
    output["respiratory"]["rule1a_arousal_stats"] = ar_stats

    # ── Step 8a (v0.6.0): LightGBM candidate re-classification ────────────
    ml_path = profile.get("ML_CLASSIFIER_PATH")
    if ml_path:
        logger.info("[pneumo 8a/10] LightGBM re-classification...")
        try:
            from .ml_classifier import apply_ml_reclassification
            cur_accepted = output["respiratory"].get("events", [])
            cur_rejected = output["respiratory"].get("rejected_hypopneas", [])
            # Compute median SpO2 + thermistor type for context features
            _spo2_for_med = spo2_data
            if _spo2_for_med is not None:
                valid = _spo2_for_med[(_spo2_for_med > 50) & (_spo2_for_med <= 100)]
                _median_spo2 = float(np.median(valid)) if len(valid) else 95.0
            else:
                _median_spo2 = 95.0
            _therm_name = ch.get("flow_thermistor", "") or ""
            _therm_type = 0 if "therm" in _therm_name.lower() else 1
            # overall_qual5 not generally available; pass 0 as neutral
            _qual5 = 0
            # Zie PostProcessingRules.ml_arousal_features: dit model is
            # getraind terwijl de arousal-features aan de inferentiekant
            # altijd nul waren (issue #16). v0.13.0 repareert die plumbing,
            # maar het model heeft nooit een niet-nulwaarde gezien. Tot het
            # opnieuw getraind is krijgt het wat het kent — dat houdt ook de
            # NSRR/paper-v31-reproductie exact.
            _ml_arousals = (arousals
                            if profile.get("ML_AROUSAL_FEATURES", False)
                            else [])
            new_acc, new_rej, ml_meta = apply_ml_reclassification(
                accepted=cur_accepted,
                rejected=cur_rejected,
                arousals=_ml_arousals,
                hypno=hypno,
                sig_dur_s=raw.times[-1],
                tst_h=sum(1 for s in hypno if s in ("N1","N2","N3","R")) * 30 / 3600.0,
                overall_qual5=_qual5,
                median_spo2=_median_spo2,
                thermistor_type=_therm_type,
                booster_path=ml_path,
                threshold=profile.get("ML_THRESHOLD", 0.65),
            )
            if ml_meta.get("status") == "ok":
                output["respiratory"]["events"]               = new_acc
                output["respiratory"]["rejected_hypopneas"]   = new_rej
                output["respiratory"]["summary"]              = _compute_summary(
                    new_acc, hypno, artifact_epochs
                )
                output["respiratory"]["ml_reclassification"]  = ml_meta
            else:
                output["respiratory"]["ml_reclassification"] = ml_meta
                logger.warning("[ml] Skipped: %s", ml_meta.get("status"))
        except Exception as e:
            logger.warning("[ml] Re-classification failed: %s", e)
            output["respiratory"]["ml_reclassification"] = {"status": f"exception: {e}"}

    # ── Step 8b0: splits lange events op fysiologische ankers ─────────────
    # Vóór de desaturatielimiet: splitsen gebeurt juist ÓP desaturaties, dus
    # elk deel heeft daarna zijn eigen anker en de limiter hoort niet te vuren.
    # Default None = geen splitsing; zie PostProcessingRules.
    _split_s = profile.get("SPLIT_EVENTS_LONGER_THAN_S")
    if _split_s is not None and output["respiratory"].get("success"):
        try:
            from .respiratory import split_long_events
            from .spo2 import detect_desaturations
            _des = []
            if spo2_data is not None:
                _sf_s = sf_spo2 or sf_flow or 1.0
                _sleep = np.zeros(len(spo2_data), dtype=bool)
                for _ep, _st in enumerate(hypno):
                    if _st in ("N1", "N2", "N3", "R"):
                        _a = int(_ep * 30.0 * _sf_s)
                        _b = int((_ep + 1) * 30.0 * _sf_s)
                        if _a < len(_sleep):
                            _sleep[_a:min(_b, len(_sleep))] = True
                if _sleep.any():
                    _des = [d["onset_s"] for d in detect_desaturations(
                        np.asarray(spo2_data, dtype=float), _sf_s, _sleep,
                        drop_pct=float(profile.get("DESATURATION_DROP_PCT", 3.0)))]
            _aro = [float(a.get("onset_s", 0.0)) for a in (arousals or [])]
            _acc, _rej, _sstats = split_long_events(
                output["respiratory"].get("events", []),
                output["respiratory"].get("rejected_hypopneas", []),
                threshold_s=float(_split_s),
                desat_onsets=_des, arousal_onsets=_aro,
            )
            if _sstats["n_split"]:
                output["respiratory"]["events"] = _acc
                output["respiratory"]["rejected_hypopneas"] = _rej
                output["respiratory"]["summary"] = _compute_summary(
                    _acc, hypno, artifact_epochs)
                logger.info(
                    "[pneumo 8b0] %d lange events gesplitst in %d delen "
                    "(%d op desaturatie, %d op arousal), %d fragmenten afgewezen",
                    _sstats["n_split"], _sstats["n_parts"],
                    _sstats["anchor_desat"], _sstats["anchor_arousal"],
                    _sstats["n_fragments_rejected"])
            output["respiratory"]["long_event_split"] = _sstats
        except Exception as e:
            logger.warning("[pneumo 8b0] splitsing overgeslagen: %s", e)
            output["respiratory"]["long_event_split"] = {"status": f"exception: {e}"}

    # ── Step 8b: begrens hergebruik van één desaturatie ───────────────────
    # Na 8a, zodat ook door de ML gepromoveerde kandidaten meetellen. Default
    # None = geen begrenzing = byte-identiek; zie PostProcessingRules.
    _desat_limit = profile.get("MAX_EVENTS_PER_DESATURATION")
    if _desat_limit is not None and output["respiratory"].get("success"):
        try:
            from .respiratory import limit_events_per_desaturation
            _acc, _rej, _dstats = limit_events_per_desaturation(
                output["respiratory"].get("events", []),
                output["respiratory"].get("rejected_hypopneas", []),
                spo2_data, sf_spo2 or sf_flow or 1.0, hypno,
                max_events=int(_desat_limit),
                drop_pct=float(profile.get("DESATURATION_DROP_PCT", 3.0)),
                post_win_s=float(profile.get("POST_EVENT_WINDOW_S", 45.0)),
            )
            if _dstats["n_degraded"]:
                output["respiratory"]["events"] = _acc
                output["respiratory"]["rejected_hypopneas"] = _rej
                output["respiratory"]["summary"] = _compute_summary(
                    _acc, hypno, artifact_epochs
                )
                logger.info(
                    "[pneumo 8b/10] desaturatiehergebruik begrensd op %d: "
                    "%d events gedegradeerd over %d groepen",
                    int(_desat_limit), _dstats["n_degraded"],
                    _dstats["n_groups_over_limit"])
            output["respiratory"]["desat_reuse_limit"] = _dstats
        except Exception as e:
            logger.warning("[pneumo 8b/10] begrenzing overgeslagen: %s", e)
            output["respiratory"]["desat_reuse_limit"] = {"status": f"exception: {e}"}

    # ── Step 9: Cheyne-Stokes ──────────────────────────────────────────────
    logger.info("[pneumo 9/10] Cheyne-Stokes detection...")
    if ref_flow is not None:
        try:
            flow_env_csr = preprocess_flow(ref_flow, sf_ref)
            output["cheyne_stokes"] = detect_cheyne_stokes(
                flow_env_csr, sf_apnea, hypno
            )
        except Exception as e:
            logger.warning("CSR detection failed: %s", e)
            output["cheyne_stokes"] = {"success": False, "csr_detected": False, "error": str(e)}
    else:
        output["cheyne_stokes"] = {"success": False, "csr_detected": False}

    # ── Fix 3: Retroactief CSR-event markering na CSR-detectie ────────────
    csr_info = output.get("cheyne_stokes", {})
    if csr_info.get("csr_detected") and output["respiratory"].get("success"):
        from .respiratory import _flag_csr_events
        events_flagged = _flag_csr_events(
            output["respiratory"]["events"], csr_info
        )
        output["respiratory"]["events"] = events_flagged
        # Herbereken summary met CSR-info
        artifact_epochs_for_summary = output["respiratory"]["summary"].get(
            "_artifact_epochs", None
        )
        output["respiratory"]["summary"] = _compute_summary(
            events_flagged,
            hypno,
            artifact_epochs,
            csr_info=csr_info,
        )
        n_flagged = sum(1 for e in events_flagged if e.get("csr_flagged"))
        logger.info("Fix3 (pipeline): %d events gemarkeerd als CSR-gerelateerd", n_flagged)

    # ── Step 9b (v0.8.16): RERA index and RDI ─────────────────────────────
    # MUST run AFTER the CSR summary recompute above: Fix 3 replaces
    # output["respiratory"]["summary"] with a fresh _compute_summary() dict on
    # CSR-positive nights, which would otherwise drop the RERA/RDI/REM-NREM keys
    # this adds. On non-CSR nights Fix 3 does not fire and nothing between the
    # old (step 8b) and this position touches the summary, so the values are
    # unchanged; on CSR nights the keys are now retained (bug fix, v0.7.4).
    logger.info("[pneumo 9b/11] RERA and RDI computation...")
    _compute_rera_rdi(output, hypno, arousals, artifact_epochs)

    # ── Step 9c: clinical phenotype flags (POSA, REM-predominant) — v0.10.0 ──
    # Output-additive; derived from the already-computed position + REM/NREM indices.
    _compute_phenotypes(output, hypno)

    # ── Step 9d: ventilatory burden (AJRCCM 2023) — v0.10.0, recalibrated v0.12.0 ──
    # VB = % of sleep with airflow < 50% of the eupneic baseline (proportion of
    # "small breaths"); bounded 0–100%, normative ≈ ≤25%. Replaces the earlier
    # %·min/h area integral.
    if ref_flow is not None and output["respiratory"].get("success"):
        try:
            _vb_fn = _compute_flow_norm(ref_flow, sf_ref)
            output["respiratory"]["summary"]["ventilatory_burden"] = (
                compute_ventilatory_burden(
                    _vb_fn, sf_ref,
                    output["respiratory"].get("_breaths", []), hypno))
        except Exception as e:  # noqa: BLE001 — VB failure must not abort scoring
            logger.warning("[pneumo] ventilatory burden failed: %s", e)

    # ── Step 10: Hypoxic burden (Azarbarzin et al., AJRCCM 2019) ──────────
    if spo2_data is not None and output["respiratory"].get("success"):
        try:
            resp_events = output["respiratory"].get("events", [])
            hb = compute_hypoxic_burden(
                spo2_data, sf_spo2, resp_events, hypno,
                local_baseline_only=bool(
                    profile.get("HYPOXIC_BURDEN_LOCAL_BASELINE", False)),
            )
            output["hypoxic_burden"] = hb
            if output.get("spo2", {}).get("summary"):
                output["spo2"]["summary"]["hypoxic_burden"] = hb.get("hypoxic_burden")
                output["spo2"]["summary"]["hypoxic_burden_unit"] = hb.get("unit")
            logger.info("[pneumo 10/10] Hypoxic burden: %.1f %%·min/h (%d events)",
                        hb.get("hypoxic_burden") or 0, hb.get("n_events_with_burden", 0))
        except Exception as e:
            logger.warning("Hypoxic burden failed: %s", e)
            output["hypoxic_burden"] = {"hypoxic_burden": None, "error": str(e)}
    else:
        output["hypoxic_burden"] = {"hypoxic_burden": None}

    # ── Step 11: Post-processing — central/mixed refinement ───────────────
    if output["respiratory"].get("success"):
        try:
            from .postprocess import postprocess_respiratory_events
            pp = postprocess_respiratory_events(
                events=output["respiratory"].get("events", []),
                csr_info=output.get("cheyne_stokes"),
                thorax_data=thorax_data,
                abdomen_data=abdomen_data,
                sf_effort=sf_apnea or sf_flow or 0,
                ahi_interval=output["respiratory"].get("summary", {}).get("ahi_interval"),
            )
            output["respiratory"]["events"] = pp["events"]
            output["postprocess"] = {
                "n_csr_reclassified": pp["n_csr_reclassified"],
                "n_mixed_decomposed": pp["n_mixed_decomposed"],
                "n_mixed_to_central": pp["n_mixed_to_central"],
                "central_instability": pp["central_instability"],
                "cai_change": pp["cai_change"],
            }
            logger.info("[pneumo 11/11] Post-processing: CSR-recl=%d, mixed-decomp=%d",
                        pp["n_csr_reclassified"], pp["n_mixed_to_central"])

            # Issue #18: de laatste _compute_summary() staat op stap 9, hierna
            # wisselen events van type. De n-kolom beschreef daardoor de
            # toestand VOOR de herklassificatie en de confidence-uitsplitsing
            # die erna — zichtbaar als een verschil dat exact gelijk is aan
            # n_csr_reclassified. Alleen de labels bewegen (ahi_total blijft
            # gelijk), maar oahi schoof op één echte opname 32,2 -> 28,6, over
            # de grens ernstig/matig heen.
            #
            # We MERGEN in plaats van te vervangen: stap 9b/9c hebben RERA,
            # RDI, REM/NREM-indices en fenotypes aan dezelfde dict toegevoegd,
            # en een kale vervanging gooide die weg (dezelfde val als in v0.7.4).
            if profile.get("SUMMARY_AFTER_RECLASSIFICATION"):
                _old = output["respiratory"].get("summary") or {}
                _fresh = _compute_summary(
                    pp["events"], hypno, artifact_epochs,
                    csr_info=output.get("cheyne_stokes"),
                )
                _moved = [k for k in ("n_obstructive", "n_central", "n_mixed",
                                      "oahi", "ahi_total")
                          if _old.get(k) != _fresh.get(k)]
                output["respiratory"]["summary"] = {**_old, **_fresh}
                logger.info("[pneumo 11/11] summary herberekend na "
                            "herklassificatie; gewijzigd: %s",
                            ", ".join(_moved) or "niets")
        except Exception as e:
            logger.warning("Post-processing failed: %s", e)
            output["postprocess"] = {"error": str(e)}

    # ── Step 12 (v0.11.0): AASM v3 clinical enrichments (output-additive) ──
    # Run last, after every summary recompute + central/CSR reclassification, so
    # the fields survive and reflect the final events. None of these touch the
    # AHI / OSAS grade (golden regression stays byte-identical).
    _compute_dual_ahi(output, profile)                 # A1: Rule 1A vs 1B (4%) AHI
    _annotate_csr_density(output, hypno)               # A2: CSR density criterion G.1(b)
    _compute_arousal_etiology(output, hypno)           # A3: resp/spont/PLM arousal indices
    _flag_apneas_at_cap(output, profile)               # A4: possible long-central-apnea truncation
    _mark_hypoventilation_not_assessed(output)         # A6: explicit scope statement

    logger.info("Pneumo analysis complete.")
    return output


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _maybe_apply_ripsum_fallback(
    hypop_flow, sf_hypop,
    thorax_data, abdomen_data, sf_rip,
    profile_dict, output,
):
    """v0.5.1: when profile requests it AND nasal pressure quality is
    poor (sensor flat / unplugged / extreme attenuation), substitute
    the thoracoabdominal RIP sum as the hypopnea-detection input.

    Quality test: per-30-second peak-to-trough excursions on the
    Pres signal; trigger fallback if median excursion is below 0.5%
    of the signal's 99th-percentile absolute amplitude (essentially
    flat) OR excursion-CV across the night is below 0.10
    (signal not modulated by breathing).

    Reference: Vaquerizo-Villar et al. DRIVEN, npj Dig Med 2024;
    Nassi et al. IEEE TBME 2022.
    """
    strategy = profile_dict.get("FLOW_FALLBACK_STRATEGY", "none")
    if strategy != "ripsum_on_nasal_failure":
        output["meta"]["flow_fallback"] = {"strategy": strategy, "triggered": False}
        return hypop_flow, sf_hypop
    if (hypop_flow is None or thorax_data is None or abdomen_data is None
            or sf_hypop is None or sf_rip is None):
        output["meta"]["flow_fallback"] = {
            "strategy": strategy, "triggered": False,
            "reason": "channels_unavailable",
        }
        return hypop_flow, sf_hypop

    try:
        seg_n  = max(1, int(sf_hypop * 30))
        n_segs = len(hypop_flow) // seg_n
        if n_segs < 5:
            output["meta"]["flow_fallback"] = {
                "strategy": strategy, "triggered": False,
                "reason": "recording_too_short",
            }
            return hypop_flow, sf_hypop
        excursions = np.array([
            float(hypop_flow[i*seg_n:(i+1)*seg_n].max()
                  - hypop_flow[i*seg_n:(i+1)*seg_n].min())
            for i in range(n_segs)
        ])
        median_exc = float(np.median(excursions))
        amp99      = float(np.percentile(np.abs(hypop_flow), 99))
        rel_exc    = (median_exc / amp99) if amp99 > 1e-9 else 0.0
        cv = (float(np.std(excursions) / median_exc)
              if median_exc > 1e-9 else 0.0)
        nasal_failed = (rel_exc < 0.005) or (cv < 0.10)
    except Exception as e:
        logger.warning("Nasal pressure quality check failed: %s", e)
        output["meta"]["flow_fallback"] = {
            "strategy": strategy, "triggered": False,
            "reason": f"qc_exception: {e!s}",
        }
        return hypop_flow, sf_hypop

    if not nasal_failed:
        output["meta"]["flow_fallback"] = {
            "strategy":            strategy,
            "triggered":           False,
            "nasal_excursion_cv":  round(cv, 3),
            "nasal_relative_excursion": round(rel_exc, 4),
        }
        return hypop_flow, sf_hypop

    # Build RIPsum at sf_hypop sample rate
    if sf_rip != sf_hypop:
        from scipy import signal as sp_signal
        n_target = int(len(thorax_data) * sf_hypop / sf_rip)
        thorax_r  = sp_signal.resample(thorax_data,  n_target)
        abdomen_r = sp_signal.resample(abdomen_data, n_target)
    else:
        thorax_r  = thorax_data
        abdomen_r = abdomen_data
    L = min(len(thorax_r), len(abdomen_r))
    ripsum = thorax_r[:L] + abdomen_r[:L]
    output["meta"]["flow_fallback"] = {
        "strategy":           strategy,
        "triggered":          True,
        "nasal_excursion_cv": round(cv, 3),
        "nasal_relative_excursion": round(rel_exc, 4),
        "fallback_signal":    "thorax+abdomen_sum",
    }
    logger.info(
        "[pneumo] RIPsum fallback: nasal-pressure quality poor "
        "(rel_exc=%.4f, cv=%.3f) → using thorax+abdomen sum as hypopnea input",
        rel_exc, cv,
    )
    return ripsum, sf_hypop


def _resolve_flow_channels(
    flow_data, sf_flow,
    flow_pressure_data, sf_fp,
    flow_therm_data, sf_ft,
    ch, output, additive_thermistor=False,
    thermistor_gate="envelope_agreement",
):
    """Assign apnea (thermistor) and hypopnea (pressure) channels per AASM.

    ``additive_thermistor`` — gebruikt het profiel de thermistor NAAST de
    neusdruk (dual-sensor) in plaats van in plaats ervan?

    ``thermistor_gate`` — welke toets beslist. "envelope_agreement" is de
    bestaande poort en de default; "respiratory_band" kijkt naar het
    thermistorkanaal alleen. Zie PostProcessingRules.thermistor_gate.
    """
    # De AASM schrijft de thermistor voor apneus voor, maar alleen een
    # thermistor die de ademhaling volgt is de juiste sensor. Op een montage
    # waar het kanaal aanwezig is maar geen bruikbaar signaal draagt, zakte
    # het aantal apneus op een echte opname van 93 naar 0 en verschoof de
    # uitkomst van matig CSAS — bevestigd door menselijke scoring — naar
    # mild SAS.
    #
    # Of die toets BLOKKEERT hangt af van hoe het profiel de thermistor
    # gebruikt, en dat onderscheid is wezenlijk:
    #
    #   VERVANGEND (aasm_v3_rec e.d.) — de thermistor wordt DE apneu-sensor
    #       en verdringt de neusdruk. Een onbruikbaar kanaal is dan
    #       destructief: het neemt events weg die er wel zijn. Hier blijft de
    #       toets blokkeren.
    #
    #   ADDITIEF (aasm_v3_dual) — beide sensoren worden gebruikt en er wordt
    #       niets afgewezen. Een onbruikbaar kanaal draagt dan hooguit nul
    #       apneus bij, wat op de echte opname ook precies gebeurde.
    #       Blokkeren voegt hier alleen risico toe, temeer omdat de drempel
    #       aantoonbaar niets voorspelt: op 25 MESA-opnames is de correlatie
    #       tussen overeenstemming en dubbele bevestiging r = +0,07, en boven
    #       en onder de drempel is dat aandeel identiek 0,13. Hier meet en
    #       rapporteert de toets alleen.
    _therm_check = None
    # Welke poort draaide, en of die BLOKKEERDE, is niet af te leiden uit het
    # getal alleen. Onder een additief profiel meet dezelfde poort wel en
    # weigert niet — een lezer die alleen `agreement: 0.31` ziet, concludeert
    # ten onrechte dat de thermistor geweigerd is. Dit legt de beslissing
    # vast in plaats van alleen de meting.
    _gate_meta = {
        "gate": ("respiratory_band" if str(thermistor_gate) == "respiratory_band"
                 else "envelope_agreement"),
        "mode": "informational" if additive_thermistor else "blocking",
        "threshold": None,
    }
    if flow_therm_data is not None and flow_pressure_data is not None:
        try:
            if str(thermistor_gate) == "respiratory_band":
                from .signal_quality import (assess_thermistor_band_power,
                                             THERMISTOR_BAND_POWER_MIN)
                _gate_meta["threshold"] = float(THERMISTOR_BAND_POWER_MIN)
                _therm_check = assess_thermistor_band_power(
                    flow_therm_data, sf_ft or 0.0)
            else:
                from .signal_quality import (assess_flow_sensor_agreement,
                                             THERMISTOR_AGREEMENT_MIN)
                _gate_meta["threshold"] = float(THERMISTOR_AGREEMENT_MIN)
                _therm_check = assess_flow_sensor_agreement(
                    flow_pressure_data, sf_fp or 0.0,
                    flow_therm_data, sf_ft or 0.0)
            if additive_thermistor and not _therm_check["usable"]:
                logger.warning(
                    "[pneumo] lage sensorovereenstemming, thermistor WEL "
                    "gebruikt (additief, wijst niets af): %s",
                    _therm_check["reason"])
            elif not _therm_check["usable"]:
                logger.warning(
                    "[pneumo] thermistor niet gebruikt als apneu-sensor: %s",
                    _therm_check["reason"])
                flow_therm_data, sf_ft = None, None
                ch = {**ch, "flow_thermistor_rejected": ch.get("flow_thermistor")}
                ch.pop("flow_thermistor", None)
            else:
                logger.info("[pneumo] thermistor bruikbaar als apneu-sensor: %s",
                            _therm_check["reason"])
        except Exception as e:  # noqa: BLE001 — nooit de run laten vallen
            logger.warning("[pneumo] thermistortoets mislukt, neusdruk "
                           "aangehouden: %s", e)
            flow_therm_data, sf_ft = None, None
            # Ook hier de afwijzing vastleggen. Dit pad liet het kanaal
            # verdwijnen zonder spoor, waarna het rapport "thermistor niet in
            # montage" toonde terwijl hij in het EDF zat en de toets alleen
            # niet uit te voeren was. Een mislukte toets is geen afwezig
            # kanaal, en de reden hoort mee: alleen een leeg veld is erger dan
            # een fout veld. Uitsluitend metadata — de scoring verandert niet.
            _therm_check = {
                "agreement": None,
                "usable":    False,
                "reason":    f"thermistortoets mislukt: {e}",
            }
            ch = {**ch, "flow_thermistor_rejected": ch.get("flow_thermistor")}
            ch.pop("flow_thermistor", None)

    if flow_pressure_data is not None or flow_therm_data is not None:
        # v0.8.4 FIX: Python 'or' crashes on numpy arrays ("truth value ambiguous").
        # Use explicit None-checks instead.
        apnea_flow = (flow_therm_data if flow_therm_data is not None
                       else flow_pressure_data if flow_pressure_data is not None
                       else flow_data)
        hypop_flow = (flow_pressure_data if flow_pressure_data is not None
                       else flow_therm_data if flow_therm_data is not None
                       else flow_data)
        sf_apnea   = sf_ft if flow_therm_data  is not None else (sf_fp or sf_flow)
        sf_hypop   = sf_fp if flow_pressure_data is not None else (sf_ft or sf_flow)
        if flow_data is None:
            flow_data = (flow_pressure_data if flow_pressure_data is not None
                         else flow_therm_data)
        output["meta"]["flow_channels"] = {
            "apnea_sensor":    ch.get("flow_thermistor")  or ch.get("flow_pressure") or ch.get("flow"),
            "hypopnea_sensor": ch.get("flow_pressure") or ch.get("flow_thermistor") or ch.get("flow"),
            "dual_sensor":     flow_pressure_data is not None and flow_therm_data is not None,
            # Zichtbaar maken waarom de thermistor er niet is: aanwezig maar
            # afgekeurd is iets anders dan afwezig, en dat verschil hoort in
            # het rapport te kunnen landen.
            "thermistor_check":   _therm_check,
            "thermistor_rejected": ch.get("flow_thermistor_rejected"),
            "thermistor_gate":    _gate_meta,
        }
    else:
        apnea_flow = flow_data
        hypop_flow = flow_data
        sf_apnea   = sf_flow
        sf_hypop   = sf_flow
        output["meta"]["flow_channels"] = {
            "apnea_sensor":    ch.get("flow", "-"),
            "hypopnea_sensor": ch.get("flow", "-"),
            "dual_sensor":     False,
            "thermistor_check":   _therm_check,
            "thermistor_rejected": ch.get("flow_thermistor_rejected"),
            "thermistor_gate":    _gate_meta,
        }
    return apnea_flow, hypop_flow, sf_apnea, sf_hypop


def _pick_eeg(raw, ch) -> tuple:
    """Selecteer het beste EEG-kanaal uit de beschikbare kanalen."""
    name = ch.get("eeg")
    if not name:
        for c in raw.ch_names:
            if any(p in c.upper() for p in ("EEG", "C3", "C4", "F3", "F4", "CZ")):
                name = c
                break
    if name and name in raw.ch_names:
        return raw.get_data(picks=[name])[0], raw.info["sfreq"]
    return None, None


def _pick_eeg_multi(raw, ch) -> list:
    """Geordende [(naam, data, sf), ...] voor de arousal-afleidingsset in ``raw``:
    centraal (== ``_pick_eeg``) + occipitaal + frontaal — enkel wat aanwezig is.
    Element 0 is de single-channel pick, zodat single-modus een strikte subset is;
    ontdubbeld op kanaalnaam."""
    data0, sf0 = _pick_eeg(raw, ch)
    if data0 is None:
        return []
    name0 = ch.get("eeg")
    if not name0:
        for c in raw.ch_names:
            if any(p in c.upper() for p in ("EEG", "C3", "C4", "F3", "F4", "CZ")):
                name0 = c
                break
    out = [(name0, data0, sf0)]
    used = {name0}

    def _find(keys):
        for k in keys:
            for c in raw.ch_names:
                if k in c.upper() and c not in used:
                    return c
        return None

    for keys in (("O2-M1", "O1-M2", "O2-A1", "O1-A2", "O2", "O1", "OZ"),   # occipitaal
                 ("F4-M1", "F3-M2", "F4-A1", "F3-A2", "F4", "F3")):         # frontaal
        nm = _find(keys)
        if nm:
            out.append((nm, raw.get_data(picks=[nm])[0], sf0))
            used.add(nm)
    return out


def _pick_emg(raw, ch) -> np.ndarray | None:
    """Selecteer het beste EMG-kanaal (kin-EMG) uit de beschikbare kanalen."""
    name = ch.get("emg")
    if not name:
        for c in raw.ch_names:
            if any(p in c.upper() for p in ("EMG", "CHIN", "MENT")):
                name = c
                break
    if name and name in raw.ch_names:
        return raw.get_data(picks=[name])[0]
    return None


def _pick_eog(raw, ch) -> np.ndarray | None:
    """Selecteer een EOG-kanaal (voor de occipitaal-EOG-reject in multi-modus)."""
    name = ch.get("eog")
    if isinstance(name, (list, tuple)):
        name = name[0] if name else None
    if not name:
        for c in raw.ch_names:
            if any(p in c.upper() for p in ("EOG", "E1-", "E2-", "LOC", "ROC")):
                name = c
                break
    if name and name in raw.ch_names:
        return raw.get_data(picks=[name])[0]
    return None


def _compute_flow_norm(flow_data, sf_flow) -> np.ndarray | None:
    """Normaliseer het flowsignaal voor amplitude-onafhankelijke analyse."""
    if flow_data is None:
        return None
    try:
        env = preprocess_flow(flow_data, sf_flow)
        bl  = compute_dynamic_baseline(env, sf_flow)
        return np.clip(env / bl, 0, 2)
    except Exception:
        return None


def _compute_phenotypes(output: dict, hypno: list) -> None:
    """v0.10.0: derive clinical phenotype flags from already-computed indices.

    - **Positional OSA (Cartwright):** supine AHI ≥ 2× non-supine AHI, with OSA
      present (AHI ≥ 5) and ≥ 30 min sleep in both the supine and non-supine groups.
      Also flags positional-therapy candidacy (non-supine AHI < 5).
    - **REM-predominant OSA:** REM-AHI ≥ 2× NREM-AHI, with ≥ 30 min REM.

    Output-additive: writes ``output["respiratory"]["summary"]["phenotypes"]``; the
    AHI and every other index are untouched.
    """
    try:
        from .constants import EPOCH_LEN_S
        summ = output.get("respiratory", {}).get("summary", {})
        if not summ:
            return
        ahi = summ.get("ahi_total")
        ph: dict = {}

        # ---- Positional OSA (Cartwright) ----
        pos = (output.get("position") or {}).get("summary", {}) or {}
        ahi_pos = pos.get("ahi_per_pos", {}) or {}
        st_min = pos.get("sleep_time_min", {}) or {}
        sup_ahi = ahi_pos.get("Supine")
        sup_min = st_min.get("Supine") or 0.0
        ns_keys = ("Left", "Right", "Prone")
        ns_min = sum((st_min.get(k) or 0.0) for k in ns_keys)
        ns_events = sum((ahi_pos.get(k) or 0.0) * ((st_min.get(k) or 0.0) / 60.0)
                        for k in ns_keys)
        ns_ahi = round(ns_events / (ns_min / 60.0), 1) if ns_min > 0 else None
        if (ahi is not None and ahi >= 5 and sup_ahi is not None and ns_ahi is not None
                and sup_min >= 30 and ns_min >= 30):
            ratio = round(sup_ahi / ns_ahi, 1) if ns_ahi > 0 else None
            flag = bool(sup_ahi >= 2 * ns_ahi) if ns_ahi > 0 else True
            ph["positional_osa"] = {
                "flag": flag,
                "ahi_supine": sup_ahi,
                "ahi_non_supine": ns_ahi,
                "supine_non_supine_ratio": ratio,
                "positional_therapy_candidate": bool(flag and ns_ahi < 5),
                "pct_supine": (pos.get("sleep_pct", {}) or {}).get("Supine"),
            }

        # ---- REM-predominant OSA ----
        rem_ahi = summ.get("rem_ahi")
        nrem_ahi = summ.get("nrem_ahi")
        rem_min = sum(1 for s in hypno if s == "R") * (EPOCH_LEN_S / 60.0)
        if (ahi is not None and ahi >= 5 and rem_ahi is not None
                and nrem_ahi is not None and nrem_ahi > 0 and rem_min >= 30):
            ph["rem_predominant"] = {
                "flag": bool(rem_ahi >= 2 * nrem_ahi),
                "rem_ahi": rem_ahi,
                "nrem_ahi": nrem_ahi,
                "rem_nrem_ratio": round(rem_ahi / nrem_ahi, 1),
                "rem_min": round(rem_min, 1),
            }

        summ["phenotypes"] = ph
    except Exception as e:  # noqa: BLE001 — phenotype derivation must not abort scoring
        logger.warning("[pneumo] phenotype computation failed: %s", e)


def _compute_rera_rdi(output: dict, hypno: list, arousals: list,
                      artifact_epochs: list | None = None) -> None:
    """Compute RERA index and RDI from FRI events + flattening + arousal coupling.

    Two RERA sources (v0.8.16):
      1. **FRI-RERA**: Flow-reduction events (≥30% amplitude drop, no ≥3% desat,
         not reinstated by Rule 1B) WITH arousal within 15s. These are the
         "classical" psgscoring RERAs from v0.8.16.
      2. **Flattening-RERA** (NEW): Breath sequences with flattening index >0.30
         (Hosselet et al., AJRCCM 1998), duration ≥10s, WITHOUT amplitude
         reduction meeting hypopnea threshold, terminated by arousal.
         These represent true upper-airway flow limitation — the "missing"
         RERA type that amplitude-only analysis cannot detect.

    RDI = AHI + RERA index.
    """
    resp = output.get("respiratory", {})
    if not resp.get("success"):
        return

    def _span(e):
        a = float(e.get("onset_s") or 0.0)
        return a, a + float(e.get("duration_s") or 0.0)

    #: Deel van de kandidaat dat binnen een al gescoord event moet vallen om
    #: hem als datzelfde event te beschouwen. Afgeleid, niet gekozen: op de
    #: golden-cases liggen de afgewezen hypopnee-kandidaten die met een event
    #: overlappen op 0,83-1,00, terwijl flattening-reeksen die een event
    #: schampen op 0,06-0,22 liggen. Daartussen zit een gat, en "meer dan de
    #: helft ligt erbinnen" is de leesbare formulering ervan.
    _SAME_EPISODE_FRACTION = 0.50

    def _overlaps(seg, others):
        """Is dit interval in hoofdzaak hetzelfde als een al geteld event?

        Hier stond een toets op ONSET-NABIJHEID: `abs(onset_a - onset_b) < 5`.
        Die mist een kandidaat die zes seconden na een hypopnee begint maar er
        volledig binnen valt, en telt omgekeerd twee dingen als hetzelfde
        wanneer ze toevallig vlak na elkaar beginnen zonder elkaar te raken.

        Elke overlap afwijzen is te grof gebleken: een flattening-reeks die
        voor 6% de rand van een hypopnee raakt is flow-limitatie NAAST dat
        event, geen duplicaat ervan. Vandaar de fractie.
        """
        a0, a1 = seg
        dur = max(a1 - a0, 1e-9)
        for b0, b1 in (_span(o) for o in others):
            if (min(a1, b1) - max(a0, b0)) / dur > _SAME_EPISODE_FRACTION:
                return True
        return False

    # ── Source 1: FRI-based RERAs (amplitude reduction + arousal) ──────
    rejected  = resp.get("rejected_hypopneas", [])
    events = resp.get("events", [])
    # Een afgewezen hypopnee die overlapt met een gescoord event is datzelfde
    # event; de exacte-onset-vergelijking die hier stond liet een verschoven
    # duplicaat door.
    fri_events = [r for r in rejected if not _overlaps(_span(r), events)]

    fri_rera_count = 0
    if arousals and fri_events:
        arousal_times = [(float(a.get("onset_s", 0)), float(a.get("duration_s", 3)))
                         for a in arousals]
        for fri in fri_events:
            fri_end = float(fri["onset_s"]) + float(fri["duration_s"])
            has_arousal = any(
                fri["onset_s"] <= a_onset <= fri_end + 15.0
                for a_onset, _ in arousal_times
            )
            if has_arousal:
                fri_rera_count += 1

    # ── Source 2: Flattening-based RERAs (flow limitation + arousal) ───
    flat_rera_count = 0
    breaths = resp.get("_breaths", [])
    if arousals and breaths:
        arousal_times = [(float(a.get("onset_s", 0)), float(a.get("duration_s", 3)))
                         for a in arousals]
        flat_seqs = _find_flattening_sequences(breaths)
        for seq in flat_seqs:
            seq_end = seq["onset_s"] + seq["duration_s"]
            # Check arousal within 15s of sequence end
            has_arousal = any(
                seq["onset_s"] <= a_onset <= seq_end + 15.0
                for a_onset, _ in arousal_times
            )
            # Al geteld als respiratoir event of als FRI-RERA? Op
            # intervaloverlap, niet op onset-nabijheid — zie _overlaps().
            _seq_span = (seq["onset_s"], seq_end)
            if (has_arousal
                    and not _overlaps(_seq_span, events)
                    and not _overlaps(_seq_span, fri_events)):
                flat_rera_count += 1

    rera_count = fri_rera_count + flat_rera_count

    # ── TST and indices ────────────────────────────────────────────────
    from .constants import EPOCH_LEN_S
    sleep_stages = {"N1", "N2", "N3", "R"}
    art_set = set(artifact_epochs or [])
    n_sleep = sum(1 for i, s in enumerate(hypno)
                  if s in sleep_stages and i not in art_set)
    # Geen ondergrens: zie psgscoring/indices.py. Hier werd de RERA-index
    # en daarmee de RDI het aantal maal duizend op een opname zonder
    # bruikbare slaaptijd.
    tst_h = n_sleep * EPOCH_LEN_S / 3600

    _ahi_raw = resp.get("summary", {}).get("ahi_total")
    rera_index = per_hour(rera_count, tst_h)
    # Zonder noemer bestaat de RERA-index niet, en dan bestaat de RDI ook niet:
    # AHI + niets is geen getal.
    rdi = (None if rera_index is None or _ahi_raw is None
           else round(float(_ahi_raw) + rera_index, 1))

    # Store in respiratory summary
    resp["summary"]["n_rera"]          = rera_count
    resp["summary"]["n_rera_fri"]      = fri_rera_count
    resp["summary"]["n_rera_flattening"] = flat_rera_count
    resp["summary"]["rera_index"]      = rera_index
    resp["summary"]["rdi"]             = rdi
    resp["summary"]["n_fri"]           = len(fri_events) - fri_rera_count

    # REM vs NREM AHI (v0.8.16)
    rem_events  = [e for e in events if e.get("stage") == "R"]
    nrem_events = [e for e in events if e.get("stage") in {"N1","N2","N3"}]
    n_rem  = sum(1 for i, s in enumerate(hypno) if s == "R" and i not in art_set)
    n_nrem = sum(1 for i, s in enumerate(hypno) if s in {"N1","N2","N3"} and i not in art_set)
    rem_h  = n_rem * EPOCH_LEN_S / 3600     # zie psgscoring/indices.py
    nrem_h = n_nrem * EPOCH_LEN_S / 3600
    resp["summary"]["rem_ahi"]  = round(len(rem_events) / rem_h, 1) if n_rem > 0 else None
    resp["summary"]["nrem_ahi"] = round(len(nrem_events) / nrem_h, 1) if n_nrem > 0 else None

    logger.info("[pneumo] RERA: %d (FRI:%d + flattening:%d), index %.1f/h; "
                "RDI: %.1f/h; REM-AHI: %s, NREM-AHI: %s",
                rera_count, fri_rera_count, flat_rera_count,
                rera_index, rdi,
                resp["summary"]["rem_ahi"], resp["summary"]["nrem_ahi"])


# ═══════════════════════════════════════════════════════════════════════════
# v0.11.0 — AASM v3 clinical enrichments (output-additive; see docs/aasm)
# ═══════════════════════════════════════════════════════════════════════════

def _ahi_severity(ahi) -> str | None:
    """AASM AHI severity band (5 / 15 / 30)."""
    if ahi is None:
        return None
    if ahi < 5:
        return "normal"
    if ahi < 15:
        return "mild"
    if ahi < 30:
        return "moderate"
    return "severe"


def _hypopnea_criterion_str(profile: dict) -> str | None:
    """A5: human-readable hypopnea scoring criterion (AASM v3 VIII.D Note 1)."""
    try:
        flow = round((1.0 - float(profile.get("HYPOPNEA_THRESHOLD", 0.70))) * 100)
        desat = float(profile.get("DESATURATION_DROP_PCT", 3.0))
        rule = str(profile.get("_AASM_RULE") or "")
        prefix = f"{rule}: " if rule[:1].isdigit() else ""
        if profile.get("DESAT_OR_AROUSAL"):
            return f"{prefix}≥{flow}% flow reduction ≥10 s + (≥{desat:g}% desaturation OR arousal)"
        if profile.get("DESAT_REQUIRED"):
            return f"{prefix}≥{flow}% flow reduction ≥10 s + ≥{desat:g}% desaturation"
        return f"{prefix}≥{flow}% flow reduction ≥10 s"
    except Exception:  # noqa: BLE001
        return None


#: Rule 1B / CMS requires a ≥4% desaturation, in percentage points — the same
#: unit as the per-event ``desaturation_pct`` field (cf. DESATURATION_DROP_PCT).
RULE_1B_DESAT_PCT = 4.0


def _rule_1b_ahi(events: list, tst_h: float) -> float | None:
    """AHI under AASM v3 Rule 1B: every apnea, plus hypopneas reaching ≥4% desat.

    Filtering the Rule 1A event list is *exact* here, not an approximation. The
    two rules share the flow-reduction (≥30%) and duration (≥10 s) criteria and
    differ only in the qualifier, and a ≥4% desaturation is by definition also a
    ≥3% one — so every Rule 1B hypopnea is already present in the Rule 1A list.
    What Rule 1B drops is exactly the hypopneas that qualified on arousal alone
    or on a 3–4% desaturation.

    The apnea set matches ``ahi_total``: type in (obstructive, central, mixed),
    with "uncertain" excluded there and therefore here too. Counting it on one
    side only would let Rule 1B exceed Rule 1A, which the rules cannot do.
    """
    if not tst_h or tst_h <= 0:
        return None
    n_apnea = n_hyp = 0
    for e in events:
        t = str(e.get("type", ""))
        if "hypopnea" in t:
            if (e.get("desaturation_pct") or 0.0) >= RULE_1B_DESAT_PCT:
                n_hyp += 1
        elif t in ("obstructive", "central", "mixed"):
            n_apnea += 1
    return (n_apnea + n_hyp) / tst_h


def _compute_dual_ahi(output: dict, profile: dict) -> None:
    """A1: expose AASM v3 hypopnea Rule 1A (≥3% desat OR arousal) and Rule 1B
    (≥4% desat) AHI side by side. Output-additive.

    Rule 1B is derived from the final, fully post-processed event list (see
    ``_rule_1b_ahi``). Rule 1A is the headline AHI when the active profile
    already scores on 3%-or-arousal; for a 4%/CMS profile the roles swap — its
    headline *is* Rule 1B — and Rule 1A comes from the ``standard`` arm of the
    robustness sweep.

    Until v0.14.1 Rule 1B was read from the sweep's ``strict`` arm instead.
    That is wrong twice over. ``aasm_v3_strict`` is a *conservative variant of
    Rule 1A* — it keeps ``desat_or_arousal`` with ``desat_threshold = 0.03``
    and differs in the stability filter, breath-level detection and nadir
    window — so no ≥4% criterion was ever applied to the number printed under a
    "≥4% desat" heading. And because the sweep re-scores on the apnea channel,
    a montage whose thermistor carries no usable signal collapsed that arm: one
    real recording read "Rule 1B 10.3/h — Mild CSAS" beside "Rule 1A 33.9/h —
    Severe CSAS" for the same night, from 78 apneas that both rules count.
    """
    try:
        summ = output.get("respiratory", {}).get("summary", {})
        if not summ:
            return
        resp = output.get("respiratory", {}) or {}
        events = resp.get("events", []) or []
        tst_h = summ.get("tst_hours")

        if profile.get("DESAT_OR_AROUSAL", True):
            ahi_1a = summ.get("ahi_total")
            # Without a successful SpO2 analysis every desaturation is None, so
            # Rule 1B would silently degrade to "apneas only" — a number that
            # looks like a result. Report nothing rather than that.
            if not (output.get("spo2") or {}).get("success"):
                return
            ahi_1b = _rule_1b_ahi(events, tst_h)
        else:
            # The active profile already requires ≥4%: its headline is Rule 1B.
            ahi_1b = summ.get("ahi_total")
            iv = output.get("ahi_interval", {}) or {}
            if "error" in iv:
                return
            ahi_1a = (iv.get("standard", {}) or {}).get("ahi")
        if ahi_1a is None or ahi_1b is None:
            return
        summ["ahi_dual"] = {
            "rule_1a": {
                "ahi": round(float(ahi_1a), 1),
                "severity": _ahi_severity(float(ahi_1a)),
                "criterion": "≥30% flow + (≥3% desat OR arousal)",
                "label": "AASM v3 Rule 1A (recommended)",
            },
            "rule_1b_4pct": {
                "ahi": round(float(ahi_1b), 1),
                "severity": _ahi_severity(float(ahi_1b)),
                "criterion": "≥30% flow + ≥4% desat",
                "label": "AASM v3 Rule 1B / CMS (4%)",
            },
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("[pneumo] dual-AHI computation failed: %s", e)


def _annotate_csr_density(output: dict, hypno: list) -> None:
    """A2: add the AASM v3 Cheyne-Stokes *density* criterion G.1(b) — ≥5 central
    apneas/hypopneas per hour of sleep over ≥2 h of monitoring — on top of the
    existing crescendo/decrescendo periodicity criterion G.1(a). Output-additive.
    """
    try:
        csr = output.get("cheyne_stokes", {})
        if not csr or csr.get("error"):
            return
        from .constants import EPOCH_LEN_S
        events = output.get("respiratory", {}).get("events", []) or []
        n_central = sum(1 for e in events
                        if e.get("type") in ("central", "hypopnea_central"))
        n_sleep_ep = sum(1 for s in hypno if s in ("N1", "N2", "N3", "R"))
        tst_h = n_sleep_ep * EPOCH_LEN_S / 3600.0
        monitoring_h = len(hypno) * EPOCH_LEN_S / 3600.0
        central_idx = round(n_central / tst_h, 1) if tst_h > 0 else 0.0
        density_ok = bool(central_idx >= 5 and monitoring_h >= 2.0)
        csr["central_events"] = n_central
        csr["central_events_per_h"] = central_idx
        csr["monitoring_hours"] = round(monitoring_h, 1)
        csr["density_criterion_met"] = density_ok             # G.1(b)
        csr["criteria_met"] = bool(csr.get("csr_detected") and density_ok)
    except Exception as e:  # noqa: BLE001
        logger.warning("[pneumo] CSR density annotation failed: %s", e)


def _compute_arousal_etiology(output: dict, hypno: list) -> None:
    """A3: surface arousal aetiology as per-hour indices (AASM V.A Note 4):
    respiratory-, spontaneous- and PLM-related arousal indices. Output-additive;
    writes into ``output["arousal"]["summary"]``.

    v0.12.0 fix: the aetiology counts (``n_respiratory_arousals`` +
    ``n_spontaneous_arousals``) and the ``arousal_index`` are derived from the *same*
    arousal set, but the index uses the artifact-corrected sleep time as denominator.
    Dividing the raw counts by a differently-computed TST made the sub-indices
    inconsistent with the total. We therefore split the ``arousal_index`` itself by
    aetiology fraction, so ``respiratory + spontaneous == arousal_index`` exactly.
    PLM-related arousals are a SUBSET of the spontaneous group ("of which PLM").
    """
    try:
        ar = output.get("arousal", {})
        summ = ar.get("summary", {})
        if not summ:
            return
        ai = summ.get("arousal_index")
        n_resp = summ.get("n_respiratory_arousals")
        n_spont = summ.get("n_spontaneous_arousals")
        if ai is None or n_resp is None or n_spont is None:
            return
        n_total = n_resp + n_spont
        if n_total <= 0:
            summ["respiratory_arousal_index"] = 0.0
            summ["spontaneous_arousal_index"] = 0.0
            return
        summ["respiratory_arousal_index"] = round(ai * n_resp / n_total, 1)
        summ["spontaneous_arousal_index"] = round(ai * n_spont / n_total, 1)
        # PLMS arousal index (subset of spontaneous): a PLM with an arousal onset
        # within -0.5 .. +3 s of the movement, expressed on the same denominator.
        plm_events = (output.get("plm", {}) or {}).get("events", []) or []
        arousals = ar.get("events", []) or []
        if plm_events and arousals:
            a_onsets = sorted(float(a.get("onset_s", 0)) for a in arousals)
            n_plm_ar = 0
            for p in plm_events:
                p0 = float(p.get("onset_s", 0))
                p1 = p0 + float(p.get("duration_s", 0) or 0)
                if any(p0 - 0.5 <= ao <= p1 + 3.0 for ao in a_onsets):
                    n_plm_ar += 1
            summ["n_plm_arousals"] = n_plm_ar
            summ["plm_arousal_index"] = round(ai * min(n_plm_ar, n_total) / n_total, 1)
    except Exception as e:  # noqa: BLE001
        logger.warning("[pneumo] arousal-etiology indices failed: %s", e)


def _flag_apneas_at_cap(output: dict, profile: dict) -> None:
    """A4: count apneas sitting at the max-duration cap (possible split of a
    genuinely long central apnea). Output-additive; no effect on AHI."""
    try:
        cap = float(os.environ.get("PSGSCORING_APNEA_MAX_DUR_S")
                    or profile.get("APNEA_MAX_DUR_S", 90.0))
        events = output.get("respiratory", {}).get("events", []) or []
        apneas = [e for e in events
                  if e.get("type") in ("obstructive", "central", "mixed", "uncertain")]
        at_cap = sum(1 for e in apneas if float(e.get("duration_s", 0) or 0) >= cap - 1.0)
        output["respiratory"]["summary"]["n_apneas_at_max_dur"] = at_cap
        output["respiratory"]["summary"]["apnea_max_dur_s"] = cap
        if at_cap:
            logger.info("[pneumo] A4: %d apnea(s) at the %.0fs duration cap "
                        "(possible split of a long central apnea)", at_cap, cap)
    except Exception as e:  # noqa: BLE001
        logger.warning("[pneumo] apnea-cap flag failed: %s", e)


def _mark_hypoventilation_not_assessed(output: dict) -> None:
    """A6: state explicitly that hypoventilation (AASM VIII.F) is out of scope —
    it requires an arterial/transcutaneous/end-tidal PCO2 channel absent from the
    standard montage. Output-additive."""
    try:
        summ = output.get("respiratory", {}).get("summary")
        if summ is not None and "hypoventilation" not in summ:
            summ["hypoventilation"] = {
                "assessed": False,
                "reason": "no arterial/transcutaneous/end-tidal PCO2 channel in montage",
            }
    except Exception as e:  # noqa: BLE001
        logger.warning("[pneumo] hypoventilation stub failed: %s", e)


def _find_flattening_sequences(breaths: list,
                                min_flat_index: float = 0.30,
                                min_dur_s: float = 10.0,
                                min_consecutive: int = 3) -> list:
    """Find consecutive breath sequences with elevated flattening index.

    AASM: flow limitation = inspiratory flattening ≥10s.
    Hosselet et al.: flattening index >0.30 indicates plateau.

    Parameters
    ----------
    breaths : list of breath dicts with 'flattening', 'onset_s', 'duration_s'
    min_flat_index : threshold for flattening (0.30 = plateau)
    min_dur_s : minimum sequence duration (AASM: ≥10s)
    min_consecutive : minimum consecutive flat breaths

    Returns
    -------
    list of {onset_s, duration_s, avg_flattening, n_breaths}
    """
    sequences = []
    run_start = None
    run_breaths = []

    for b in breaths:
        fi = b.get("flattening")
        if fi is not None and fi >= min_flat_index:
            if run_start is None:
                run_start = float(b["onset_s"])
            run_breaths.append(b)
        else:
            # End of run — check if it qualifies
            if run_breaths and len(run_breaths) >= min_consecutive:
                last = run_breaths[-1]
                dur = float(last["onset_s"]) + float(last["duration_s"]) - run_start
                if dur >= min_dur_s:
                    sequences.append({
                        "onset_s": run_start,
                        "duration_s": round(dur, 1),
                        "avg_flattening": round(float(np.mean([
                            b["flattening"] for b in run_breaths
                            if b.get("flattening") is not None
                        ])), 3),
                        "n_breaths": len(run_breaths),
                    })
            run_start = None
            run_breaths = []

    # Handle trailing run
    if run_breaths and len(run_breaths) >= min_consecutive:
        last = run_breaths[-1]
        dur = float(last["onset_s"]) + float(last["duration_s"]) - run_start
        if dur >= min_dur_s:
            sequences.append({
                "onset_s": run_start,
                "duration_s": round(dur, 1),
                "avg_flattening": round(float(np.mean([
                    b["flattening"] for b in run_breaths
                    if b.get("flattening") is not None
                ])), 3),
                "n_breaths": len(run_breaths),
            })

    return sequences


def _parse_edf_patient_info(raw) -> dict:
    """Extract patient info from EDF/EDF+ header via MNE Raw object.

    EDF+ Patient ID (80 bytes at offset 8):
      patient_code  sex  birthdate  patient_name
      e.g.: "MCH-0234567 F 02-MAY-1951 Haagse_Harry"

    EDF+ Recording ID (80 bytes at offset 88):
      Startdate dd-MMM-yyyy admincode techniciancode equipmentcode

    Fields NOT in EDF: weight, height, BMI (require manual entry).
    """
    from datetime import datetime

    info = {
        "patient_code": None, "sex": None, "birthdate": None,
        "birthday_str": None, "name": None, "recording_date": None,
        "admin_code": None, "technician": None, "equipment": None,
    }

    # Get file path from MNE raw
    fpath = None
    if hasattr(raw, "filenames") and raw.filenames:
        fpath = raw.filenames[0]
    elif hasattr(raw, "_filenames") and raw._filenames:
        fpath = raw._filenames[0]
    if not fpath:
        return info

    try:
        with open(fpath, "rb") as f:
            f.seek(8)
            patient_raw = f.read(80).decode("latin-1").strip()
            recording_raw = f.read(80).decode("latin-1").strip()
    except (OSError, UnicodeDecodeError):
        return info

    # Parse EDF+ patient ID
    parts = patient_raw.split()
    if len(parts) >= 4:
        info["patient_code"] = parts[0] if parts[0] != "X" else None
        info["sex"] = parts[1] if parts[1] in ("M", "F") else None
        if parts[2] != "X":
            try:
                info["birthdate"] = datetime.strptime(parts[2], "%d-%b-%Y").isoformat()
                info["birthday_str"] = parts[2]
            except ValueError:
                info["birthday_str"] = parts[2]
        name_parts = parts[3:]
        raw_name = " ".join(name_parts).replace("_", " ")
        info["name"] = raw_name if raw_name and raw_name != "X" else None
    elif len(parts) >= 1 and patient_raw and patient_raw != "X":
        info["name"] = patient_raw.replace("_", " ")

    # Parse EDF+ recording ID
    rparts = recording_raw.split()
    if len(rparts) >= 2 and rparts[0] == "Startdate":
        try:
            info["recording_date"] = datetime.strptime(rparts[1], "%d-%b-%Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
        if len(rparts) >= 3:
            info["admin_code"] = rparts[2] if rparts[2] != "X" else None
        if len(rparts) >= 4:
            info["technician"] = rparts[3].replace("_", " ") if rparts[3] != "X" else None
        if len(rparts) >= 5:
            info["equipment"] = rparts[4].replace("_", " ") if rparts[4] != "X" else None

    logger.info("[pneumo] EDF patient: %s, sex=%s, dob=%s, equipment=%s",
                info.get("name"), info.get("sex"),
                info.get("birthday_str"), info.get("equipment"))
    return info


def _empty_arousal(reason: str) -> dict:
    """Geeft een leeg arousal-resultaat terug met foutmelding."""
    return {
        "success": False,
        "error":   reason,
        "summary": {
            "arousal_index": None,
            "n_respiratory_arousals": None,
            "n_spontaneous_arousals": None,
            "pct_respiratory_arousals": None,
            "n_reras": 0,
            "rera_index": 0,
            "rdi": None,
            "clinical_interpretation": [],
        },
    }
