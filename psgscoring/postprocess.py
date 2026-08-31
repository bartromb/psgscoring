"""
psgscoring.postprocess
======================
Post-processing module for refining respiratory event classification.

Implements:
1. CSR-aware central reclassification — events in CSR troughs are
   reclassified as central regardless of effort signal artifacts.
2. Mixed apnea decomposition — mixed events with central portion ≥10 s
   are reclassified as central.
3. Central instability index — quantifies profile-dependent uncertainty
   in obstructive vs. central classification.

Added in v0.3.0.

References
----------
Berry RB et al. AASM Manual v2.6, 2020.
Azarbarzin A et al. AJRCCM 2019;200(2):211-219.
"""

from __future__ import annotations

import logging
import numpy as np
from .utils import safe_r

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CSR-aware central reclassification
# ---------------------------------------------------------------------------

def reclassify_csr_events(
    events: list,
    csr_info: dict,
    confidence_floor: float | None = None,
) -> list:
    """
    Reclassify CSR-flagged events as central.

    When Cheyne-Stokes respiration is detected, events in the trough of
    the crescendo-decrescendo cycle are physiologically central — even if
    the effort signal shows apparent paradoxical breathing (often cardiac
    pulsation artifact in heart failure patients).

    Events with ``csr_flagged=True`` that are currently classified as
    ``"obstructive"`` or ``"mixed"`` are reclassified as ``"central"``
    with the original type preserved in ``original_type``.

    G3 (audit-trail rollback): the ``original_type`` field is preserved
    so downstream consumers (manual review, audit trail) can revert a
    CSR-driven reclassification by restoring ``original_type`` if the
    CSR detection is later deemed false-positive. This is the v0.4.4
    interim mechanism; v0.5 will add a full append-only audit log.

    Parameters
    ----------
    events : list[dict]
        Respiratory events (already CSR-flagged by _flag_csr_events).
    csr_info : dict
        CSR detection result from detect_cheyne_stokes().

    Returns
    -------
    list[dict] — modified events with reclassifications applied.
    """
    if not csr_info or not csr_info.get("csr_detected"):
        return events

    n_reclassified = 0
    modified = []

    for ev in events:
        ev = dict(ev)
        if ev.get("csr_flagged") and ev.get("type") in ("obstructive", "mixed"):
            ev["original_type"] = ev["type"]
            ev["type"] = "central"
            ev["csr_reclassified"] = True
            # GEEN confidence-verhoging meer, tenzij een profiel erom vraagt.
            #
            # Hier stond `max(conf, 0.80)`, met als toelichting "CSR context
            # provides good evidence". Die aanname is op 14-08-2026 gemeten en
            # weerlegd: op CSR-nachten haalt de obstructief-versus-centraal
            # beslissing kappa 0,091 tegen 0,311 zonder CSR, en de dominante
            # fout is juist obstructief -> centraal (230 van de 235). De regel
            # verhoogde dus het vertrouwen in precies de beslissing die daar
            # het zwakst is. Zie docs/subtypering_mesa_20260814.md.
            #
            # Dat is zichtbaar in het rapport: de sterrenkolom leest de
            # confidence, en 0,80 valt in de band 0,60-0,84. Op een opname met
            # 29 herclassificaties stonden er 26 op twee sterren, uitsluitend
            # omdat deze regel dat getal zette.
            #
            # Het event behoudt nu zijn eigen confidence uit
            # `classify_apnea_type`. `csr_reclassified` blijft erop staan, dus
            # wie de herkomst wil zien, ziet hem.
            if confidence_floor is not None:
                ev["confidence"] = max(ev.get("confidence", 0.5),
                                       float(confidence_floor))
            ev["classify_detail"] = ev.get("classify_detail", {})
            if isinstance(ev["classify_detail"], dict):
                ev["classify_detail"]["csr_reclassified"] = True
            n_reclassified += 1
        modified.append(ev)

    if n_reclassified > 0:
        logger.info(
            "CSR reclassification: %d events reclassified as central", n_reclassified
        )

    return modified


# ---------------------------------------------------------------------------
# 2. Mixed apnea decomposition
# ---------------------------------------------------------------------------

def decompose_mixed_apneas(
    events: list,
    thorax_data: np.ndarray | None,
    abdomen_data: np.ndarray | None,
    sf_effort: float,
    central_threshold_s: float = 10.0,
) -> list:
    """
    Decompose mixed apneas into central and obstructive portions.

    For each mixed apnea, the effort signal (thorax + abdomen) is analysed
    to determine the duration of the central portion (absent effort) and
    the obstructive portion (resumed effort against closed airway).

    G2 assumption (AASM-conform): the central phase is at the START of
    the event, the obstructive phase is at the END. This matches the
    canonical AASM mixed-apnea pattern. Heterogeneous apneas with
    interleaved effort phases (clinically unusual) will only have their
    leading low-effort block counted as the central portion; later
    intra-event effort gaps are not separately accounted for. If a
    population is encountered where this matters, the algorithm here
    needs to be extended to scan for ANY contiguous low-effort window
    ≥ central_threshold_s.

    If the central portion ≥ central_threshold_s, the event is
    reclassified as ``"central"`` with ``mixed_decomposed=True``.

    Parameters
    ----------
    events : list[dict]
        Respiratory events from the pipeline.
    thorax_data, abdomen_data : ndarray or None
        Raw effort signals.
    sf_effort : float
        Sampling frequency of effort signals.
    central_threshold_s : float
        Minimum central portion duration (seconds) to reclassify as
        central. Default 10.0 s (per AASM guideline).

    Returns
    -------
    list[dict] — modified events with decomposition metadata.
    """
    if thorax_data is None and abdomen_data is None:
        return events

    # Use sum of thorax + abdomen for effort detection
    if thorax_data is not None and abdomen_data is not None:
        # Align lengths
        min_len = min(len(thorax_data), len(abdomen_data))
        effort = np.abs(thorax_data[:min_len]) + np.abs(abdomen_data[:min_len])
    elif thorax_data is not None:
        effort = np.abs(thorax_data)
    else:
        effort = np.abs(abdomen_data)

    n_decomposed = 0
    n_reclassified = 0
    modified = []

    for ev in events:
        ev = dict(ev)

        if ev.get("type") != "mixed":
            modified.append(ev)
            continue

        onset_s = float(ev.get("onset_s", 0))
        dur_s = float(ev.get("duration_s", 0))
        if dur_s < 3:
            modified.append(ev)
            continue

        # Extract effort segment for this event
        idx_start = int(onset_s * sf_effort)
        idx_end = int((onset_s + dur_s) * sf_effort)
        idx_end = min(idx_end, len(effort))

        if idx_start >= idx_end:
            modified.append(ev)
            continue

        seg = effort[idx_start:idx_end]

        # Determine effort threshold: amplitude < 20% of segment max
        # indicates absent effort (central portion)
        # NaN-safe: a single NaN in the effort segment would otherwise make
        # np.max → NaN, silently defeating the central-portion detection.
        seg_max = np.nanmax(seg) if np.any(np.isfinite(seg)) else 0.0
        if not np.isfinite(seg_max) or seg_max < 1e-10:
            # Entire event has no effort → pure central
            ev["central_duration_s"] = safe_r(dur_s)
            ev["obstructive_duration_s"] = 0.0
            ev["central_ratio"] = 1.0
            ev["mixed_decomposed"] = True
            ev["original_type"] = "mixed"
            ev["type"] = "central"
            ev["confidence"] = max(ev.get("confidence", 0.5), 0.85)
            n_reclassified += 1
            n_decomposed += 1
            modified.append(ev)
            continue

        effort_threshold = 0.20 * seg_max
        low_effort = seg < effort_threshold

        # Find the central portion: contiguous low-effort from event start
        # (AASM: mixed apnea starts central, transitions to obstructive)
        central_samples = 0
        for val in low_effort:
            if val:
                central_samples += 1
            else:
                break

        central_dur = central_samples / sf_effort
        obstr_dur = dur_s - central_dur

        ev["central_duration_s"] = safe_r(central_dur, 1)
        ev["obstructive_duration_s"] = safe_r(obstr_dur, 1)
        ev["central_ratio"] = safe_r(
            central_dur / dur_s if dur_s > 0 else 0, 2
        )
        ev["mixed_decomposed"] = True
        n_decomposed += 1

        # Reclassify if central portion dominates
        if central_dur >= central_threshold_s:
            ev["original_type"] = "mixed"
            ev["type"] = "central"
            ev["confidence"] = max(ev.get("confidence", 0.5), 0.80)
            n_reclassified += 1

        modified.append(ev)

    if n_decomposed > 0:
        logger.info(
            "Mixed decomposition: %d analysed, %d reclassified as central",
            n_decomposed, n_reclassified,
        )

    return modified


# ---------------------------------------------------------------------------
# 3. Central instability index
# ---------------------------------------------------------------------------

def compute_central_instability_index(
    ahi_strict: float | None,
    ahi_standard: float | None,
    ahi_sensitive: float | None,
    oahi_strict: float | None = None,
    oahi_standard: float | None = None,
    oahi_sensitive: float | None = None,
) -> dict:
    """
    Quantify the uncertainty in obstructive vs. central classification
    by comparing OAHI across scoring profiles.

    A high instability index indicates many ambiguous events where the
    central vs. obstructive nature depends on scoring stringency.

    Parameters
    ----------
    ahi_* : float or None
        Total AHI for each profile.
    oahi_* : float or None
        Obstructive AHI for each profile (if available).

    Returns
    -------
    dict with keys:
        central_instability_index : float (0-1 scale)
        interpretation : str
        ahi_range : float (max - min AHI across profiles)
    """
    result = {
        "central_instability_index": None,
        "interpretation": "insufficient data",
        "ahi_range": None,
    }

    # Use OAHI if available, otherwise AHI
    vals = []
    if oahi_strict is not None and oahi_sensitive is not None:
        vals = [v for v in [oahi_strict, oahi_standard, oahi_sensitive] if v is not None]
    elif ahi_strict is not None and ahi_sensitive is not None:
        vals = [v for v in [ahi_strict, ahi_standard, ahi_sensitive] if v is not None]

    if len(vals) < 2:
        return result

    val_range = max(vals) - min(vals)
    val_mean = np.mean(vals)
    # Normalise: range / mean → coefficient of variation-like metric
    cii = val_range / val_mean if val_mean > 1.0 else val_range / 5.0

    # Clip to [0, 1]
    cii = min(1.0, max(0.0, cii))

    if cii < 0.15:
        interp = "low — classification stable across profiles"
    elif cii < 0.40:
        interp = "moderate — some events are profile-sensitive"
    else:
        interp = "high — many ambiguous events, consider manual review"

    result["central_instability_index"] = safe_r(cii, 3)
    result["interpretation"] = interp
    result["ahi_range"] = safe_r(val_range, 1)

    return result


# ---------------------------------------------------------------------------
# 4. Master post-processing function
# ---------------------------------------------------------------------------

def csr_therapy_contradiction(
    events: list,
    split_night: dict | None,
    *,
    min_diagnostic_ahi: float = 15.0,
    max_residual_ahi: float = 5.0,
    min_reclassified_share: float = 0.50,
) -> dict | None:
    """Spreekt de therapiehelft de CSR-herclassificatie tegen?

    Bij een split-night levert de tweede helft een onafhankelijk signaal dat de
    scoring niet gebruikt: hoe de events op DRUK reageren. Verdwijnt een
    ernstige AHI vrijwel volledig onder CPAP, dan waren die events
    drukresponsief -- kenmerkend voor obstructieve ziekte. Centrale apneu
    reageert daar niet zo op.

    Staat er dan in hetzelfde rapport dat de meerderheid van de centrale events
    uit een CSR-HERCLASSIFICATIE komt (obstructief/gemengd -> centraal), dan
    bevat het rapport twee uitspraken die niet samengaan. Deze functie zegt dat
    hardop in plaats van het aan de lezer over te laten.

    Waarom dit ertoe doet: `docs/subtypering_mesa_20260814.md` meet die
    herclassificatie op MESA n=52 en vindt kappa 0,091 op CSR-nachten (0,311
    zonder), met 230 van de 235 fouten in de richting obstructief -> centraal.
    De stap is dus zwak op precies de nachten waar hij vuurt, en de
    therapierespons is het enige onafhankelijke tegenwicht dat er ligt.

    Drempels, conventioneel en niet zelfbedacht:
      * ``max_residual_ahi`` 5/u -- de gangbare grens voor behandelsucces;
      * ``min_diagnostic_ahi`` 15/u -- pas vanaf matig-ernstig is een
        vrijwel-volledige respons informatief;
      * ``min_reclassified_share`` 0,50 -- de meerderheid van de centrale
        events moet uit de herclassificatie komen, anders gaat het over
        werkelijk gedetecteerde centrale events en zegt de tegenspraak niets.

    Returns ``None`` als er niets te melden valt, anders een dict met de
    getallen waarop de melding rust. Dit is een OBSERVATIE: de functie
    herclassificeert niets terug en verandert geen enkele index.
    """
    sn = split_night or {}
    if not sn.get("detected"):
        return None
    segs = sn.get("segments") or {}
    diag, ther = segs.get("diagnostic") or {}, segs.get("therapeutic") or {}
    if not (diag.get("reliable") and ther.get("reliable")):
        return None
    a_diag, a_ther = diag.get("ahi"), ther.get("ahi")
    if a_diag is None or a_ther is None:
        return None
    if not (float(a_diag) >= min_diagnostic_ahi
            and float(a_ther) < max_residual_ahi):
        return None

    centraal = [e for e in events if str(e.get("type")) == "central"]
    if not centraal:
        return None
    herklas = [e for e in centraal if e.get("csr_reclassified")]
    aandeel = len(herklas) / len(centraal)
    if aandeel < min_reclassified_share:
        return None

    return {
        "diagnostic_ahi": round(float(a_diag), 1),
        "therapeutic_ahi": round(float(a_ther), 1),
        "n_central": len(centraal),
        "n_central_from_csr_reclassification": len(herklas),
        "reclassified_share": round(aandeel, 3),
        "message": (
            "De therapiehelft spreekt de CSR-herclassificatie tegen: de AHI "
            f"zakt van {float(a_diag):.1f} naar {float(a_ther):.1f} onder "
            f"therapie, wat op drukresponsieve (obstructieve) events wijst, "
            f"terwijl {len(herklas)} van de {len(centraal)} centrale events "
            "uit een CSR-herclassificatie van obstructief/gemengd komen. Die "
            "herclassificatie is op CSR-nachten zwak gemeten (kappa 0,09). "
            "Controleer de subtypering vóór een uitspraak over centrale "
            "slaapapneu."),
    }


def postprocess_respiratory_events(
    events: list,
    csr_info: dict | None = None,
    thorax_data: np.ndarray | None = None,
    abdomen_data: np.ndarray | None = None,
    sf_effort: float = 0,
    ahi_interval: dict | None = None,
    csr_reclassification: bool = True,
    csr_confidence_floor: float | None = None,
) -> dict:
    """
    Run all post-processing refinements on respiratory events.

    Call this after the main pipeline has completed CSR flagging.

    Returns
    -------
    dict with keys:
        events : list — refined events
        n_csr_reclassified : int
        n_mixed_decomposed : int
        n_mixed_to_central : int
        central_instability : dict
        cai_standard : float — standard CAI
        cai_decomposed : float — CAI after decomposition
    """
    result = {
        "events": events,
        "n_csr_reclassified": 0,
        "n_mixed_decomposed": 0,
        "n_mixed_to_central": 0,
        "central_instability": {},
    }

    original_events = events

    # Count original central events
    n_central_before = sum(
        1 for e in events if e.get("type") == "central"
    )

    # Step 1: CSR-aware reclassification
    if csr_info and csr_reclassification:
        events = reclassify_csr_events(events, csr_info,
                                       confidence_floor=csr_confidence_floor)
        result["n_csr_reclassified"] = sum(
            1 for e in events if e.get("csr_reclassified")
        )

    # Step 2: Mixed apnea decomposition
    if sf_effort > 0:
        events = decompose_mixed_apneas(
            events, thorax_data, abdomen_data, sf_effort,
        )
        result["n_mixed_decomposed"] = sum(
            1 for e in events if e.get("mixed_decomposed")
        )
        result["n_mixed_to_central"] = sum(
            1 for e in events
            if e.get("mixed_decomposed") and e.get("type") == "central"
        )

    # Step 3: Central instability index
    if ahi_interval:
        std = ahi_interval.get("standard", {})
        strict = ahi_interval.get("strict", {})
        sensitive = ahi_interval.get("sensitive", {})
        result["central_instability"] = compute_central_instability_index(
            ahi_strict=strict.get("ahi"),
            ahi_standard=std.get("ahi"),
            ahi_sensitive=sensitive.get("ahi"),
        )

    # Count final central events
    n_central_after = sum(
        1 for e in events if e.get("type") == "central"
    )

    result["events"] = events
    result["cai_change"] = n_central_after - n_central_before

    logger.info(
        "Post-processing: CAI change %+d (CSR: %d, mixed decomp: %d)",
        result["cai_change"],
        result["n_csr_reclassified"],
        result["n_mixed_to_central"],
    )

    return result


# ---------------------------------------------------------------------------
# Twee flowsensoren: apneus samenvoegen zonder dubbel te tellen
# ---------------------------------------------------------------------------
# Twee sensoren gebruiken is NIET hetzelfde als hun events optellen. De
# fysiologie is asymmetrisch:
#
#   apneu op de NEUSDRUK terwijl de thermistor nog flow ziet
#       -> mondademhaling. Er beweegt lucht; dit is geen apneu. Precies
#          hierom schrijft de AASM de thermistor voor apneus voor.
#
#   apneu op de THERMISTOR terwijl de neusdruk nog flow ziet
#       -> fysiologisch vreemd. De thermistor voelt oraal en nasaal, dus
#          nasale flow hoort hij te zien. Wijst op thermistorartefact.
#
#   beide sensoren zien het
#       -> apneu, en met hogere zekerheid dan elk kanaal alleen kan geven.
#
# Een OF-regel zou dus juist de valse apneus toevoegen waar de AASM tegen
# beschermt. De tweede sensor hoort te FALSIFIEREN, niet op te tellen.
#
# Ontdubbelen gebeurt op OVERLAP, niet op gelijke tijden: temperatuur volgt
# trager dan druk, dus dezelfde gebeurtenis krijgt verschoven grenzen.
# Dezelfde IoU-logica als de validatieharness.

def _iou(a0, a1, b0, b1):
    """Intersection-over-union van twee tijdsintervallen."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def corroborate_apnea_events(thermistor_events, pressure_events,
                             corroboration_licensed=False,
                             iou_thresh=0.20,
                             keep_thermistor_only=None,
                             keep_pressure_only=None):
    """Kruiscontroleer apneus van twee flowsensoren.

    ``thermistor_events``  apneus gedetecteerd op de oronasale thermistor
    ``pressure_events``    apneus gedetecteerd op de nasale druk

    Elk event valt in een van drie vakjes, en het vakje bepaalt het oordeel:

    ``both``            beide sensoren zien het -> apneu, hoogste zekerheid
    ``thermistor_only`` alleen de thermistor -> verdacht artefact
    ``pressure_only``   alleen de neusdruk -> vermoedelijk mondademhaling

    **De richting van de twijfel is niet symmetrisch, en de default volgt de
    veilige kant.** Falsifieren met een sensor die je niet vertrouwt is hoe
    een echte opname 83 centrale apneus verloor. Daarom:

    ``corroboration_licensed=False`` (default)
        Er wordt NIETS afgewezen. Beide lijsten worden samengevoegd en alleen
        ontdubbeld. Bij twijfel over de tweede sensor blijft een apneu dus
        gewoon een apneu.

    ``corroboration_licensed=True``
        Alleen te geven wanneer de signaalkwaliteitstoets de tweede sensor
        betrouwbaar acht. Dan pas mogen de "only"-categorieen wegvallen.

    ``keep_thermistor_only`` / ``keep_pressure_only`` overschrijven dat per
    categorie; None betekent "volg de licentie".

    Retourneert ``(events, diagnostiek)``. Elk behouden event draagt
    ``corroboration`` met het vakje waarin het viel.
    """
    if keep_thermistor_only is None:
        keep_thermistor_only = not corroboration_licensed
    if keep_pressure_only is None:
        keep_pressure_only = not corroboration_licensed
    def _span(e):
        a = float(e.get("onset_s") or 0.0)
        return a, a + float(e.get("duration_s") or 0.0)

    therm = list(thermistor_events or [])
    press = list(pressure_events or [])
    matched_p = [False] * len(press)
    out, n_both = [], 0

    for t in therm:
        ta, tb = _span(t)
        best_j, best_iou = None, 0.0
        for j, pr in enumerate(press):
            if matched_p[j]:
                continue
            pa, pb = _span(pr)
            v = _iou(ta, tb, pa, pb)
            if v >= iou_thresh and v > best_iou:
                best_j, best_iou = j, v
        if best_j is not None:
            matched_p[best_j] = True
            e = dict(t)                       # de AASM-sensor bepaalt de grenzen
            e["corroboration"] = "both"
            out.append(e)
            n_both += 1
        elif keep_thermistor_only:
            e = dict(t)
            e["corroboration"] = "thermistor_only"
            out.append(e)

    n_t_only = sum(1 for t in therm) - n_both
    n_p_only = sum(1 for m in matched_p if not m)

    if keep_pressure_only:
        for j, pr in enumerate(press):
            if not matched_p[j]:
                e = dict(pr)
                e["corroboration"] = "pressure_only"
                out.append(e)

    out.sort(key=lambda e: float(e.get("onset_s") or 0.0))
    return out, {
        "n_thermistor": len(therm),
        "n_pressure": len(press),
        "n_both": n_both,
        "n_thermistor_only": n_t_only,
        "n_pressure_only": n_p_only,
        "n_kept": len(out),
        "iou_thresh": iou_thresh,
        "corroboration_licensed": corroboration_licensed,
        "keep_thermistor_only": keep_thermistor_only,
        "keep_pressure_only": keep_pressure_only,
    }


def merge_apnea_events(primary, secondary, iou_thresh=0.20,
                       keep="longest", secondary_only=True):
    """Voeg apneus van twee sensoren samen zonder dubbel te tellen.

    ``primary``   events van de AASM-sensor voor apneus (thermistor)
    ``secondary`` events van de andere sensor (nasale druk)
    ``keep``      welk event overblijft bij overlap: "primary", "longest"
                  of "confident" (hoogste confidence)
    ``secondary_only``
                  neem ook events op die ALLEEN op de tweede sensor te zien
                  zijn. Klinisch is dat de vraag of een apneu die alleen de
                  thermistor ziet een echt event is (mondademhaling) of een
                  artefact (losgeschoten neusbril). Staat daarom als knop,
                  niet als aanname.

    Retourneert ``(events, diagnostiek)``.
    """
    def _span(e):
        a = float(e.get("onset_s") or 0.0)
        return a, a + float(e.get("duration_s") or 0.0)

    def _better(x, y):
        if keep == "primary":
            return x
        if keep == "confident":
            return x if (x.get("confidence") or 0) >= (y.get("confidence") or 0) else y
        xa, xb = _span(x); ya, yb = _span(y)
        return x if (xb - xa) >= (yb - ya) else y

    out = [dict(e) for e in (primary or [])]
    used = [False] * len(secondary or [])
    n_merged = 0

    for i, s in enumerate(secondary or []):
        sa, sb = _span(s)
        best_j, best_iou = None, 0.0
        for j, p in enumerate(out):
            pa, pb = _span(p)
            v = _iou(sa, sb, pa, pb)
            if v >= iou_thresh and v > best_iou:
                best_j, best_iou = j, v
        if best_j is not None:
            winner = dict(_better(out[best_j], s))
            winner["seen_on_both_sensors"] = True
            out[best_j] = winner
            used[i] = True
            n_merged += 1

    n_added = 0
    if secondary_only:
        for i, s in enumerate(secondary or []):
            if not used[i]:
                e = dict(s)
                e["secondary_sensor_only"] = True
                out.append(e)
                n_added += 1

    out.sort(key=lambda e: float(e.get("onset_s") or 0.0))
    return out, {
        "n_primary": len(primary or []),
        "n_secondary": len(secondary or []),
        "n_merged": n_merged,
        "n_secondary_only_added": n_added,
        "n_total": len(out),
        "iou_thresh": iou_thresh,
        "keep": keep,
        "secondary_only": secondary_only,
    }
