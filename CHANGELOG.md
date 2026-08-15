> **Language policy.** Entries before 2026-08-14 are partly in Dutch; from
> here on, English. The v0.17.0 entry carries an English translation below the
> original because it underpins the MESA figures in the paper; earlier entries
> are deliberately left as they are rather than retranslated.

# v0.18.0 — 2026-08-15 — overview

**No profile changes its output.** All fifteen profiles score identically to
0.17.0; golden is 9/9 unchanged and the PSG-IPA figures reproduce to the
decimal (bias +1.69/h, MAE 1.76/h, r 0.997, weighted κ 0.839). This release
adds capability behind switches that are off, and records six measurements —
four of which argued *against* the change they were testing.

| what | field | default | measured outcome |
|---|---|---|---|
| Coherence-based thermistor gate | `thermistor_gate="breath_coherence"` | off (`envelope_agreement`) | a real repair of a real defect, but **costs 2.95/h of AHI bias on MESA**; default reverted |
| Long-event splitting | `split_events_longer_than_s` | `None` | no hypopnoea candidates; overlaps an existing splitter |
| Low-baseline desaturation relaxation | `desat_low_baseline_relaxation` | `False` | fires on 1.8% of events; a deviation from Rule 1A, not a repair |
| Breath-granular event boundaries | `event_boundaries="breath"` | off (`envelope`) | nearest-edge snapping does not remove the 2–5 s onset offset |
| Desaturation re-use limit | `max_events_per_desaturation` | `None` | removes 8 events of 4303 |
| Central/obstructive subtyping, remeasured | — | — | κ = 0.114; 230 of 235 obstructive-called-central errors sit on Cheyne-Stokes nights |

Four ideas evaluated from an external system were measured and **none
adopted**; `docs/third_party_comparison.md` carries one row each, including
the rejections. That document is the audit trail for what was borrowed and
what was not.

The switches ship rather than being deleted so the negative results stay
reproducible and a future cohort can overturn them. A reader who wants the
reasoning rather than the summary should read the six sections below, which
are the original measurement notes.

---

# v0.18.0 — 2026-08-15 — the thermistor gate asks an amplitude question about a timing problem

**The defect.** `assess_flow_sensor_agreement` correlates the amplitude
ENVELOPES of nasal pressure and thermistor at 1 Hz and admits the thermistor as
an apnoea sensor above 0.40. Envelope correlation is the wrong quantity. A
thermistor is a thermal sensor whose response saturates and does not track
ventilation linearly; nasal pressure before linearisation scales roughly as the
square of flow. The two therefore have legitimately different amplitude
dynamics even when both track the same breaths faithfully, and a thermistor
with correct timing but compressed dynamic range fails a criterion it should
pass.

The question the gate needs to answer is whether the two sensors observe the
SAME BREATHS — a timing question, not an amplitude one.

**The replacement.** Magnitude-squared coherence between the two channels
across the breathing band (0.10–0.50 Hz), power-weighted. Coherence is
invariant to a constant amplitude ratio between the channels — exactly the
nuisance the envelope criterion is confounded by — while remaining sensitive to
whether the two share a consistent phase relationship at the breathing
frequency. It is scale-free for the same reason the repaired effort gate is.

**DECISION RULE, declared before calibration.** The threshold is placed at the
midpoint of the gap between channel pairs that demonstrably do NOT share
breathing and pairs that demonstrably do. The negative end is constructed, not
selected: the same recording's thermistor time-reversed, and phase-randomised
surrogates preserving the power spectrum. Both retain the amplitude statistics
and destroy the timing correspondence, so they isolate the quantity being
measured.

The threshold is NOT tuned on any outcome — not AHI, not event F1, not
agreement with a reference, not the pass rate. If the two distributions
overlap so that no gap exists, that is reported and no threshold is chosen; the
existing gate then stays and the finding is that coherence does not separate
them either.

## Calibrated

Three kinds of negative, from easy to strict:

| negative | max coherence |
|---|---|
| thermistor time-reversed | 0.003 |
| phase-randomised surrogate | 0.004 |
| **thermistor from another recording** (n = 132 pairs) | **0.008** |
| — gap — | |
| real pairs, n = 25 | 0.026 – 0.771 (median 0.482) |

The first two turned out to be too easy: constructed from the same signal,
they destroy coherence completely. The cross-recording pair is the realistic
case — two sensors that do not belong to the same patient — and that is the
boundary that counts. Midpoint of that gap: $(0.008 + 0.026)/2 = 0.017$.

**The margin is narrow, and we say so.** One of the 25 recordings has a
thermistor that genuinely tracks poorly (0.026), so between "real but weak"
and "unrelated" there is a factor of three. Enough to place a threshold, not
enough to be comfortable.

## What it changes

On the same 25 MESA recordings the envelope criterion admits the thermistor on
11 (44 %); coherence admits 24 (96 %). The envelope criterion was rejecting
sensors that demonstrably track the same breathing — which is what it was
supposed to be testing for.

That the coherence threshold rejects almost nothing is the finding, not a
defect in the calibration. The decision rule forbade tuning on the pass rate,
and following it produced a number that says: on this cohort, the thermistor
nearly always sees the breathing.

## Default ON for the clinical profiles

User decision, 2026-08-14. Eleven profiles move to `breath_coherence`.
`mesa_shhs` and `chicago_1999` stay on `envelope_agreement` — this gate decides
whether apnoeas are scored on the thermistor or on nasal pressure, so it moves
the AHI, and those two reproduce published figures. The two band-power dual
profiles keep their own gate: that criterion asks the single-channel question,
and moving it too would conflate two changes.

## A bug the existing suite caught

The first implementation admitted PURE WHITE NOISE as a thermistor. The
regression came from `test_flow_reference.py`, whose fixture carries a
deliberately dead thermistor: under the new gate `aasm_v3_rec` began scoring
apnoeas on it and the ventilatory burden fell from 58.4 to 0.0.

The cause is a property of the statistic, not of the signal. Magnitude-squared
coherence is biased upward when few averaging windows are available: for K
independent segments the null expectation is about 1/K rather than zero. The
ten-minute fixture yields fifteen segments and a noise floor near 0.04, while
an eight-hour night yields hundreds and lands near 0.006. **The threshold
therefore depended on recording duration** — the same class of defect as the
two gates repaired earlier in this release, where a threshold tracked something
other than what it claimed to measure.

The estimate is now bias-corrected, `(C − 1/K)/(1 − 1/K)`, which makes it
comparable across durations. Noise on the ten-minute fixture drops from 0.037
to 0.012 and is rejected. A regression test pins that.

Two margins are stated rather than smoothed. One of the 25 recordings has a
thermistor that genuinely tracks poorly (0.024), so real-but-weak and unrelated
are separated by a factor of four. And the correction does not remove duration
dependence entirely: at ten minutes, noise still reaches 0.012 against a
threshold of 0.015. On clinical recordings of several hours the margin is
comfortable; on short fragments this number should be read with suspicion.

## Scope

New value `thermistor_gate="breath_coherence"`, now the default for the
clinical profiles.

## AHI impact on MESA: the flip makes agreement WORSE

Measured after the fact, n = 40, paired, reference `aasm15`, two runs differing
only in `--thermistor-gate`.

| profile | gate | F1 | precision | recall | bias | MAE | events |
|---|---|---|---|---|---|---|---|
| `aasm_v3_rec` | envelope | 0.441 | 0.557 | 0.396 | **−5.18** | 8.37 | 4277 |
| | coherence | 0.366 | 0.480 | 0.324 | **−8.13** | 9.24 | 3577 |
| `aasm_v3_breath` | envelope | 0.482 | 0.645 | 0.458 | **−5.33** | 9.04 | 4303 |
| | coherence | 0.470 | 0.630 | 0.420 | **−6.63** | 9.49 | 3978 |

Everything moves the wrong way: 700 fewer events on the cascade, bias from
−5.18 to −8.13, F1 −0.075, MAE up. **18 of 40 recordings changed AHI** (median
|Δ| 4.65/h, max 23.40), so this is not noise from a handful of cases.

The control holds: forcing the old gate reproduces the existing figures exactly
(`breath` bias −5.33, MAE 9.04), so the comparison isolates the gate.

The mechanism is legible. The gate blocks under mono profiles, so where it was
closed the apnoeas were scored on NASAL PRESSURE. Opening it moves them to the
THERMISTOR, which is slower and finds fewer apnoeas. The NSRR reference was
scored by human scorers on the flow channels as they used them.

**What this does not show** is that the envelope criterion was right. It
demonstrably measures the wrong quantity — a thermistor with correct timing but
compressed dynamic range fails it — and that finding stands. The correct
reading is that the old gate did the right thing for the wrong reason: on this
cohort nasal pressure is the better apnoea sensor, and rejecting the thermistor
often happened to be right.

**Recommendation: revert the default** to `envelope_agreement` and keep the
field, the repaired criterion and this measurement. The coherence gate answers
the sensor question correctly; it should be enabled only on a cohort where the
thermistor is demonstrably the better apnoea sensor, which MESA is not.

Measurement: `docs/mesa_gate_{envelope_agreement,breath_coherence}.{json,log}`. This decides on every recording whether apnoeas are
scored on the thermistor or on nasal pressure, so it moves the AHI; `mesa_shhs`
and `chicago_1999` stay pinned regardless.

# v0.18.0 — 2026-08-15 — block 2B measured against the reference: the limit stays None

MESA n=40, paired, `aasm_v3_breath`, reference `aasm15`. Three runs differing
only in `max_events_per_desaturation`.

| limit | F1 | precision | recall | bias | MAE | events |
|---|---|---|---|---|---|---|
| **None** | 0.482 | 0.645 | 0.458 | **−5.33** | 9.04 | 4303 |
| 2 | 0.479 | 0.645 | 0.455 | −5.37 | 9.08 | 4295 |
| 3 | 0.482 | 0.645 | 0.458 | −5.33 | 9.04 | 4303 |

**Limit 3 removes nothing at all** — every measure identical to three decimals.
**Limit 2 removes 8 of 4303 events (0.19 %)** and makes everything marginally
worse: F1 −0.004, bias 0.04 further from zero.

The decision rests on the argument fixed before the measurement, not on these
numbers: the limiter can only REMOVE events, while MESA already under-counts at
a bias of −5.33/h and PSG-IPA falls inside the scorer range on 5 of 5
recordings. There is no shortage of strictness for this knob to fix. The
measurement only confirms it — CAISR's hard limit of 2 costs eight events here
and returns nothing for them.

Consistent with what was already on record: on PSG-IPA limit 2 is a complete
no-op (largest group = 2), and on MESA 6 of 30 recordings had a group of 3 or
more. That translates into eight events out of 4303.

**Default stays `None`.** The field remains available and is proposed for no
profile.

Measurement: `docs/mesa_2b_{none,nonelim2,nonelim3}.{json,log}`.

## Block 2 closed: four borrowed ideas, none adopted

| idea | outcome |
|---|---|
| 2A arousal coupling window 15 → 25 s | not adopted; every widening costs precision while recall stands still |
| 2B desaturation re-use limit | not adopted; limit 2 costs 8 events, limit 3 does nothing |
| 2D split long events | built; turns out to be a different ANCHOR for an existing splitter, and 0 of 9880 hypopnoeas are even candidates |
| 2E 2 % under a low baseline | built; the specified condition was vacuous (466/466), repaired to fire on 1.8 % |

That none of them was switched on is itself the result. CAISR's parameter
values are random-search fits on their own pre-processing; ours are the AASM
thresholds themselves. That they do not transfer to these cohorts strengthens
the traceability argument of §7.4 rather than weakening it.

# v0.18.0 — 2026-08-15 — block 2D measured on the duration distribution: it overlaps an existing splitter

Before any sweep, two things that change what 2D can be.

**psgscoring already splits long events.** `_split_long_region` (v0.8.22) cuts
a region longer than the profile maximum at its point of highest flow
amplitude — the best partial recovery — recursively. The maxima are 90 s for
apnoeas and 60 s for hypopnoeas. So 2D is not a missing capability; it is a
DIFFERENT ANCHOR for a mechanism that exists. Existing: ventilation recovery.
2D: the physiological consequence, a desaturation or an arousal.

**The duration distribution has no natural gap.** MESA n=150, 16 839 events on
`aasm_v3_breath`: median 21.4 s, p99 81.1 s, a smooth decay and then a hard
ceiling at 90 s — the ceiling is the existing splitter, not physiology. The
specification allows moving the threshold to a natural gap if the measurement
shows one. It does not show one, so the threshold stays at CAISR's 60 s and is
documented as inherited rather than derived.

**What is left to split, by family:**

| family | events | above 60 s |
|---|---|---|
| hypopnoea | 9 880 | **0 (0.00 %)** |
| apnoea | 6 959 | 593 (8.52 %) |

Hypopnoeas are already capped at 60 s by the existing splitter, so at threshold
60 s block 2D can only touch apnoeas in the 60–90 s band.

**Expectation, recorded before the sweep** (as the specification requires):
event count rises on CSR nights, AHI rises on that subgroup, and almost nothing
changes elsewhere. Given the family split above, the effect is bounded from the
start: at most 593 of 16 839 events (3.5 %) are even candidates, all of them
apnoeas.


# v0.18.0 — 2026-08-15 — block 3.1 remeasured with the gate open: subtyping is weak, and it is CSR

No code change. MESA n=60 (52 usable), `aasm_v3_rec`, seed 20260801, single
arousal derivation.

On 2026-08-12 this could not be measured: the RIP gate failed 52 of 52
recordings and all 973 matched apnoeas came out `uncertain`. With the repaired
gate: **52 of 52 usable** (50 bilateral, 2 single-channel), 4 still `uncertain`
(0.4 %).

**Detection is unchanged to three decimals** — F1 median 0.222, precision
0.153, recall 0.549, identical to the earlier run. That is the control, not a
coincidence: the gate touches typing, not detection.

## Classification, 973 matched events

| ref \ algo | obstructive | central | mixed | uncertain |
|---|---|---|---|---|
| obstructive | **609** | 235 | 1 | 4 |
| central | 67 | **57** | 0 | 0 |

Accuracy 0.687, **Cohen kappa 0.114**, recall obstructive 0.721, central 0.460.

The accuracy misleads: 849 of 973 events are obstructive, so calling everything
obstructive already scores 0.87. Kappa says what accuracy hides — agreement
barely above chance, and central apnoeas are wrong more often than right.

## Stratified, the error is concentrated

| subset | n | accuracy | kappa | recall obstr. |
|---|---|---|---|---|
| all usable | 52 | 0.687 | 0.114 | 0.721 |
| with Cheyne-Stokes | 38 | 0.652 | 0.091 | 0.681 |
| without Cheyne-Stokes | 14 | **0.916** | **0.311** | **0.951** |

Of the 235 obstructive-called-central errors, **230 fall on CSR nights** and 5
outside them. That is physiologically legible: under Cheyne-Stokes the effort
amplitude itself waxes and wanes, so an obstructive event in the trough of the
cycle looks effort-poor.

This does NOT show the algorithm is wrong. Obstructive-versus-central is hard
for human scorers on CSR nights too, and the NSRR annotation is a single
scoring without a spread. The measurement locates the divergence; it does not
adjudicate it.

## The ECG-effort branch never fired

The specification asks for a stratum of recordings where it fires. Measured: 0
of 52 recordings, 0 events reclassified. The axis does not exist on this cohort
— itself a finding, since a branch that never engages on 52 MESA recordings
contributes nothing there and its effect cannot be measured.

Report: `docs/subtypering_mesa_20260814.md`.

# v0.18.0 — 2026-08-15 — block 1B step 2 measured: nearest-edge snapping does NOT fix the offset

`event_boundaries="breath"` is implemented and stays default off. Measured on
PSG-IPA against the same twelve scorers, `aasm_v3_rec`, single arousal
derivation:

| recording | hypopnoea onset | hypopnoea offset | median F1 | AHI |
|---|---|---|---|---|
| SN1 | −3.30 → **−3.51** | −5.00 → −4.99 | 0.470 → 0.470 | 8.1 → 8.1 |
| SN2 | −5.30 → **−5.75** | +9.60 → +8.50 | 0.317 → 0.333 | 9.3 → 9.3 |
| SN3 | −5.15 → −4.24 | −1.10 → −0.70 | 0.886 → 0.886 | 53.8 → 53.8 |
| SN4 | −3.15 → **−3.66** | +1.65 → +1.85 | 0.286 → 0.286 | 4.3 → 4.3 |
| SN5 | −2.10 → −2.04 | −0.30 → −1.05 | 0.349 → **0.338** | 11.0 → 11.0 |

**The onset offset does not shrink.** It improves on two recordings, worsens on
three, the median F1 moves by at most 0.016 in either direction, and the AHI is
unchanged everywhere.

**Why, and it should have been foreseen.** Nearest-edge snapping is
direction-agnostic: it moves a boundary to whichever breath edge is closest, at
most half a breath (~2 s). The measured lag is 2-5 s and systematic, so the
nearest edge to an early envelope crossing is usually the same early one. To
correct a one-directional lag you have to snap to the first *reduced* breath at
onset and the first *recovered* breath at offset -- the semantic rule, not the
geometric one. That needs a per-breath reduction verdict, which the cascade
detector does not compute. The graded detector does, which is precisely why it
already lands at half the offset.

**Consequence.** The field stays, default `"envelope"`, because the
implementation is correct for what it does and the audit trail
(`classify_detail.envelope_onset_s`) is useful on its own. It is proposed for
no profile. The envelope-lag diagnosis from the 1B measurement stands; this
geometric correction is simply not the fix, and adding a shift constant to
force it would be fitting on five recordings -- explicitly ruled out by the
block 1B specification.

Measurement: `docs/event_boundaries_psgipa_20260814.{json,log}`.

# v0.18.0 — 2026-08-15 — block 1B measurement: event boundaries

No behavioural change. `scripts/measure_boundary_offsets.py` (6 tests) measures
the SIGNED onset and offset difference of every matched algorithm-to-scorer
pair on PSG-IPA, against the human-to-human distribution of the same quantity.
Without that reference "the algorithm is 1.8 s off" cannot be interpreted,
because scorers differ among themselves too.

Convention: d = algorithm - scorer, so negative means the algorithm starts or
ends EARLIER. Run with `PSGSCORING_AROUSAL_DERIVATION=single`, matcher
IoU >= 0.20, type-agnostic.

**Sanity check first.** Human medians are +0.10, -0.05, +0.10, -0.10, -0.11 s
across 26 000 scorer pairs — no systematic offset between humans, as it should
be. The human interquartile range is the noise floor everything below is judged
against.

## Result

| recording | human IQR (onset) | `rec` hypopnoea onset | `breath` hypopnoea onset | apnoea onset |
|---|---|---|---|---|
| SN1 | −0.9 .. +1.0 | **−3.30** | −1.55 | +0.80 |
| SN2 | −1.6 .. +1.3 | **−5.30** | −2.30 | −0.10 |
| SN3 | −1.3 .. +1.8 | **−5.15** | −1.27 | −0.40 |
| SN4 | −2.6 .. +1.8 | **−3.15** | −2.38 | — |
| SN5 | −2.3 .. +1.2 | −2.10 | −2.66 | +0.55 |

**Apnoeas are fine.** Onset medians +0.80, −0.10, −0.40, +0.55 all sit inside
the human IQR; offsets sit inside on three of four. Identical between the two
profiles, as expected — apnoeas come from the same detector.

**Cascade hypopnoeas carry a systematic offset.** `aasm_v3_rec` starts
hypopnoeas 2–5 s earlier than the human scorers, in the same direction on all
five recordings, four of them outside the human IQR. This is the
"systematic offset, fixed direction" case: envelope lag. The envelope crosses
its threshold on the declining flank, while a scorer marks the first clearly
reduced breath.

**The graded detector already lands close.** `aasm_v3_breath` halves the onset
offset and puts its hypopnoea OFFSETS inside the human IQR on all five
recordings (+0.21, +0.55, −0.31, +1.00, +0.52). That is corroboration rather
than a separate finding: a breath-granular detector is exactly the correction
the envelope-lag diagnosis predicts, and it demonstrably helps.

**Reclassifying part of the recall gap.** `n_lost_to_iou` — reference events
that DO have an overlapping algorithm event but fall below the match threshold
— totals 63 for `rec` against 8 for `breath`. Eight times fewer "missed"
events on the graded detector are in fact only differently delimited.

## What follows, and what does not

Step 2 (`event_boundaries="breath"`) is NOT implemented. It moves durations,
therefore counts, therefore the AHI and every coupling window, and the golden
baseline would break by design. It needs a decision first, and on
`aasm_v3_breath` the case is weak because that detector's boundaries are
already near-human.

Bearing on block 2A: the arousal coupling window anchors on the event END.
For `aasm_v3_breath` — the profile the 2A sweep ran on — the hypopnoea offsets
sit inside the human IQR on all five recordings, so that anchor is sound and
the 2A conclusion (keep 15 s) is unaffected.

Measurement: `docs/boundary_offsets_20260814_{summary.json,pairs.csv}`.


# v0.17.0 — 2026-08-13 — de RIP-kwaliteitspoort mat de eenhedendeclaratie

## Het defect

`assess_rip_channel` keurde een effortkanaal af op twee ABSOLUTE drempels:

```python
MAD_FAILED_BELOW    = 0.005
ENERGY_FAILED_BELOW = 0.001
```

EDF-eenheden zijn per kanaal vrij. Wie RIP in mV declareert komt na de
omrekening naar V ~150x onder die drempel binnen met een volstrekt normaal
signaal; wie `n/a` declareert komt er duizenden malen boven uit. De drempel
selecteerde dus op hoe het opnamesysteem zijn eenheid opschrijft.

Gemeten, drie cohorten, drie conventies:

| cohort | eenheid | MAD | ademfractie | poort |
|---|---|---|---|---|
| MESA (n=52) | mV | 0,00003 – 0,00005 | 0,37 – 0,58 | **unreliable**, 0/52 door |
| kliniek `89e63920` | mV | 0,0016 / 0,051 | 0,64 / 0,75 | **single-channel** |
| PSG-IPA `SN1` | n/a | 164 / 233 | 0,83 / 0,91 | bilateral |

MAD spant zeven ordes van grootte; de ademfractie — de werkelijke
signaalkwaliteit — een factor 2,5. De volgorde komt niet eens overeen.

Gevolg op MESA: 52 van 52 opnames `unreliable`, dus 100 % van de apneus komt
uit als `uncertain` en de apneu-subtypering was op dat cohort niet te meten.
Klinisch valt op `89e63920` het thoraxkanaal af met de béste signaalvorm van
alles wat gemeten is, waarna de opname stil naar `single-channel` degradeert —
zonder paradoxale fasedetectie, precies de grootheid voor obstructief/centraal.

Dat dit op PSG-IPA nooit opviel, is geen toeval: dat cohort declareert `n/a`.
Het validatiecohort van de paper is juist het cohort waar de poort niet bijt.

## De reparatie

`rip_shape_metrics()` levert twee schaalvrije grootheden:

- **ademfractie** = vermogen in 0,10–0,50 Hz gedeeld door 0,02–4,0 Hz. Een
  verhouding van twee vermogens uit hetzelfde signaal, dus onafhankelijk van de
  eenheid. Witte ruis geeft ongeveer de bandbreedteverhouding (~0,10).
- **vlakke fractie** = aandeel identieke opeenvolgende monsters. Een
  losgeraakte band geeft een vlakke lijn, en dat is te zien zonder te weten hoe
  groot een normale uitslag is.

Achter `rip_quality_scale_free`, default `False` = bestaand gedrag.

## BESLISREGEL — vooraf vastgelegd, vóór de kalibratie

De faaldrempel op de ademfractie wordt gelegd **midden in het gat** tussen een
aantoonbaar dood kanaal (vlakke lijn en witte ruis, waarvan de verwachte
ademfractie analytisch ~0,10 is) en de waargenomen echte kanalen.

De drempel wordt NIET afgesteld op een uitkomstmaat — niet op AHI-bias, niet op
event-F1, niet op overeenstemming met NSRR. Dat zou de poort op de uitkomst
fitten die hij hoort te bewaken.

Is er geen gat, dan wordt dat gerapporteerd en wordt er geen getal gekozen.

## Gekalibreerd — het gat is breed

| soort | ademfractie |
|---|---|
| vlakke lijn | 0,000 |
| kwantisatieruis | 0,102 |
| witte ruis | 0,103 |
| 50 Hz-net + ruis | 0,102 |
| drift zonder ademhaling | 0,174 |
| — **gat** — | |
| echte kanalen, n=20 over drie cohorten | **0,371 – 0,912** |

Midden in het gat: (0,174 + 0,371) / 2 = 0,2725 → **`BREATH_FRACTION_FAILED_BELOW = 0.27`**.

Het dode uiteinde is niet gefit maar analytisch bekend: witte ruis verdeelt zijn
vermogen gelijkmatig, dus de ademfractie landt op de bandbreedteverhouding
(0,50 − 0,10) / (4,0 − 0,02) = 0,101. Gemeten 0,103. Daar staat een aparte test
op, want juist dat maakt de drempel verdedigbaar zonder naar een uitkomstmaat
te kijken.

`BREATH_FRACTION_WEAK_BELOW = 0.35` ligt net onder het zwakste werkelijk
waargenomen kanaal. `weak` is alleen een waarschuwing — de modus blijft
bilateraal — dus die grens heeft geen gedragsgevolg.

## Wat het doet

`mesa-sleep-1691`, `aasm_v3_rec`:

| | poort uit | poort aan |
|---|---|---|
| `recommended_mode` | `unreliable` | `bilateral` |
| uncertain | 42 | 0 |
| obstructief | 0 | 42 |
| `ahi_total` | 32,2 | **35,9** |
| `ahi_incl_uncertain` | 39,6 | 35,9 |

De twee indices vallen aan de aan-kant samen, zoals hoort wanneer er niets meer
onbepaald is. Cohortbreed (n=20): alle 20 opnames van `unreliable` naar
`bilateral` (1x single-channel), 713 `uncertain`-apneus worden 458 obstructief
/ 247 centraal / 8 gemengd — exact hetzelfde aantal — en `ahi_total` stijgt
gemiddeld +4,75/u. Zie `docs/rip_gate_effect_mesa_20260812.{json,log}`
(`scripts/effect_rip_gate_mesa.py`).

## Tegen de NSRR-referentie

`validate_mesa.py --n 40 --seed 20260801`, referentie `aasm15`, twee runs die
alleen in `--rip-scale-free` verschillen:

| profiel | poort | F1 | precisie | recall | bias | MAE | r | severity |
|---|---|---|---|---|---|---|---|---|
| `aasm_v3_rec` | uit | 0,447 | 0,508 | 0,447 | −9,48 | 10,00 | 0,763 | 20/40 |
| `aasm_v3_rec` | **aan** | 0,442 | 0,557 | 0,396 | **−5,14** | **8,37** | **0,806** | **24/40** |
| `aasm_v3_breath` | uit | 0,482 | 0,645 | 0,458 | −11,93 | 12,98 | 0,723 | 12/40 |
| `aasm_v3_breath` | **aan** | 0,482 | 0,645 | 0,458 | **−5,33** | **9,04** | **0,801** | **18/40** |

Op `aasm_v3_breath` zijn F1, precisie en recall tot op drie decimalen identiek
terwijl de bias halveert. Dat is het mechanisme, niet een toevalligheid: die
detector labelt hypopneus altijd kaal `"hypopnea"`, dus daar verandert de
detectie niet en wordt er alleen opgehouden apneus als `uncertain` uit
`ahi_total` te laten vallen. Zelfde events, andere boekhouding.

Op `aasm_v3_rec` gaat bovendien het stabiele-ademhalingsfilter draaien (zie
`stability_filter_all_hypopnea_subtypes`): precisie omhoog, recall omlaag, F1
vlak. Op beide profielen verbetert élke index-maat.

**Ruim de helft van de gerapporteerde MESA-onderdetectie was dus geen
detectieprobleem.** De drempel 0,27 is niet op deze uitkomst gekozen — hij lag
vast op de signaalvorm voordat deze meting draaide, dus dit is een
onafhankelijke bevestiging en geen fit. n = 40, één seed.

## Beide vlaggen staan default AAN op 13 van de 15 profielen

Gebruikersbesluit 13-08-2026. `rip_quality_scale_free` en
`stability_filter_all_hypopnea_subtypes` hebben nu dataclass-default `True`,
zodat een nieuw profiel de reparatie erft. Expliciet gepind op `False`:

- **`mesa_shhs`** — draagt de reproductie van paper v31/v37.
- **`chicago_1999`** — reproduceert de historische criteria van 1999.

Alle overige profielen gaan mee, inclusief `aasm_v2_rec`, `aasm_v1_rec` en
`cms_medicare`: de RIP-poort is infrastructuur, geen scoringsregel. AASM v1/v2
en de CMS-definitie zeggen niets over hoe je een dode RIP-band herkent.

### De golden was blind voor deze hele klasse fouten

Het omzetten van beide vlaggen op 13 profielen liet **alle zeven bestaande
golden-cases bit-identiek** (89 toevoegingen, 0 verwijderingen in de baseline).
Dat is geen geruststelling maar een gat: elke bestaande case heeft
effortkanalen met MAD ~0,6 en zit dus toevallig in het bereik waar een absolute
amplitudedrempel werkt — precies zoals PSG-IPA, het cohort waarop dit defect
nooit opviel.

Nieuwe case `mv_scale_effort` (`effort_scale=1e-5`, dezelfde signaalvorm in
mV-schaal) sluit dat gat, en laat de bug in zijn zuiverste vorm zien:

| poort | events | `ahi_total` |
|---|---|---|
| uit | 5 | **0,0** |
| aan | 5 | **31,6** |

Vijf apneus gedetecteerd, AHI gerapporteerd als nul — enkel omdat de
effortkanalen in mV gedeclareerd staan.

## Waarom dit een vlag is en geen stille reparatie

Kale `uncertain` valt BUITEN `ahi_total` (wel in `ahi_incl_uncertain`). Een
werkende poort zet op MESA `uncertain`-apneus om in getypeerde apneus, en dus
stijgt `ahi_total`. Daarmee is dit geen typeringswijziging maar een
INDEXwijziging, en breekt `mesa_shhs` byte-identiteit — de reproductie van
paper v31/v37.

`mesa_shhs` en `chicago_1999` blijven daarom gepind op het absolute gedrag.

Bijwerking die apart gemeten hoort: op MESA is de AHI-bias −11 tot −15/u
(onderdetectie). Als een deel van die onderdetectie apneus zijn die in
`uncertain` belandden en daar uit `ahi_total` vielen, dan verbetert de poort de
bias. Dat is een hypothese, geen claim — en ze wordt gemeten NA de kalibratie,
niet ervoor, zodat ze de drempel niet kan sturen.

---

## English translation (v0.17.0)

**The defect.** `assess_rip_channel` rejected an effort channel on two
ABSOLUTE thresholds, `MAD < 0.005` and `breath_energy < 0.001`. EDF units are
per-channel free. A recording that declares RIP in mV arrives ~150x below
those thresholds after conversion to volts, carrying a perfectly normal
signal; one that declares `n/a` arrives thousands of times above them. The
threshold therefore selected on how the acquisition system writes its unit,
not on the sensor.

Measured across three cohorts, three conventions:

| cohort | unit | MAD | breath fraction | gate |
|---|---|---|---|---|
| MESA (n=52) | mV | 0.00003 – 0.00005 | 0.37 – 0.58 | **unreliable**, 0/52 pass |
| clinical `89e63920` | mV | 0.0016 / 0.051 | 0.64 / 0.75 | **single-channel** |
| PSG-IPA `SN1` | n/a | 164 / 233 | 0.83 / 0.91 | bilateral |

MAD spans seven orders of magnitude; the breath fraction — the actual signal
quality — spans a factor of 2.5, and the two do not even rank the channels the
same way. On MESA this made 52 of 52 recordings `unreliable`, so every apnoea
came out as `uncertain` and apnoea subtyping could not be measured at all. On
the clinical recording the thorax channel is rejected despite having the best
waveform of anything measured here, after which the recording degrades
silently to `single-channel` — without paradoxical phase detection, the very
quantity obstructive/central typing rests on.

That this never showed on PSG-IPA is no coincidence: that cohort declares
`n/a`. The paper's validation cohort is precisely the cohort where the gate
does not bite.

**The repair.** `rip_shape_metrics()` returns two scale-free quantities: the
breath fraction (power in 0.10–0.50 Hz over 0.02–4.0 Hz) and the flat
fraction (share of identical consecutive samples). Both are ratios within the
same signal and therefore independent of the declared unit.

**Decision rule, declared before calibration.** The failure threshold is
placed in the middle of the gap between a demonstrably dead channel (flat line
and white noise, whose expected breath fraction is analytically ~0.10) and the
observed real channels. It is NOT tuned on any outcome measure — not AHI bias,
not event F1, not agreement with NSRR. Had there been no gap, that would have
been reported and no number chosen.

Calibration: flat line 0.000; quantisation noise 0.102; white noise 0.103;
50 Hz mains 0.102; drift without breathing 0.174; **gap**; real channels,
n=20 across three cohorts, 0.371 – 0.912. Midpoint (0.174 + 0.371)/2 = 0.2725
→ `BREATH_FRACTION_FAILED_BELOW = 0.27`. That noise lands on the bandwidth
ratio (0.50−0.10)/(4.0−0.02) = 0.101 is separately tested: the dead end of the
scale is known analytically rather than fitted.

**Why this is a flag and not a silent fix.** Bare `uncertain` falls OUTSIDE
`ahi_total` (it is counted in `ahi_incl_uncertain`). A working gate turns
`uncertain` apnoeas into typed apnoeas on MESA and therefore raises the AHI.
This is an INDEX change, not merely a typing change, and it breaks byte
identity for `mesa_shhs` — the reproduction of paper v31/v37. `mesa_shhs` and
`chicago_1999` are pinned to the absolute behaviour.

**Against the NSRR reference** (`validate_mesa.py`, n=40, reference `aasm15`,
two runs differing only in `--rip-scale-free`): `aasm_v3_rec` bias
−9.48 → −5.14, MAE 10.00 → 8.37, r 0.763 → 0.806, severity 20/40 → 24/40;
`aasm_v3_breath` bias −11.93 → −5.33, MAE 12.98 → 9.04, r 0.723 → 0.801,
severity 12/40 → 18/40. On `breath`, F1, precision and recall are identical to
three decimals while the bias halves — the same events, different accounting.
Confirmed at n=150 on 2026-08-14: bias −11.20 → −5.30 (`rec`),
−13.25 → −5.18 (`breath`), −15.02 → −2.34 (`breath_dual`).

**Both flags default ON for 13 of 15 profiles** (user decision, 2026-08-13).
`mesa_shhs` and `chicago_1999` stay pinned. The golden harness was blind to
this entire class of defect: flipping both flags left all seven existing cases
bit-identical, because every fixture has effort channels with MAD ~0.6 and
therefore sits in the range where an absolute threshold happens to work. The
new case `mv_scale_effort` (`effort_scale=1e-5`) closes that gap and shows the
bug in its purest form: five apnoeas detected, `ahi_total` 0.0 with the gate
off against 31.6 with it on.

# v0.16.0 — 2026-08-12 — the square-root linearisation depended on the montage, not on the profile

## The asymmetry

`_setup_hypop_channel` applies the AASM Rule 3 square-root linearisation only
when the hypopnoea channel is a *different* channel from the apnoea channel:

| cohort | channels | shared channel | linearised |
|---|---|---|---|
| PSG-IPA | one flow channel | yes | **no** |
| MESA | `Pres` + `Therm` | no | **yes** |

Without linearisation a true 50 % flow reduction measures as a 75 % amplitude
reduction. Reductions are systematically overstated, more candidates clear the
30 % criterion, and `hypopnea_strictness = 0.50` is calibrated on exactly that
convention. Applied to a linearised cohort the same operating point is
therefore too strict. This is the leading candidate for the opposite bias
directions observed across the two cohorts: **+1.77/h on PSG-IPA against
−11 to −15/h on MESA**.

## Added — `hypopnea_force_linearisation` (default `False`)

Makes the choice a profile decision instead of a montage property. Default is
current behaviour: every shipped profile is byte-identical, golden harness
8/8 unchanged, 686 tests green.

When enabled on a shared channel, the envelope is recomputed with
`is_nasal_pressure=True` **and the precomputed baseline is discarded**. That
second half matters: the precomputed baseline lives on the non-linearised
scale, so reusing it would divide a linearised numerator by a non-linearised
denominator, and the ratio would no longer measure a flow reduction. The run
records which branch executed in `hypopnea_linearised`,
`hypopnea_channel_shared` and `hypopnea_linearisation_forced`.

## Decision rule, declared BEFORE the sweep

Enabling linearisation and re-deriving `hypopnea_strictness` are **one
experiment, not two**. The v0.14.0 entry did only the first half — linearisation
on, operating point unchanged — observed bias +1.77 → −3.15 and MAE 1.84 → 3.15,
and concluded that the twelve corrections are calibrated against non-linearised
behaviour. That is precisely what a smaller measured reduction produces at
unchanged strictness; the conclusion does not follow from the measurement.

Therefore, stated here in advance and not after seeing the results:

> **`hypopnea_strictness` will be re-derived as the point where the AHI bias
> against the PSG-IPA scorer median crosses zero.** Not the F1 maximum — that
> would be fitting on the outcome measure. The sweep runs 0.30–0.60 in steps of
> 0.025. If two points bracket zero, the one with the smaller |bias| is taken;
> if they tie, the lower strictness (more inclusive) is taken, because the
> measured failure mode on MESA is under-detection.

Measurement plan, in this order: (1) PSG-IPA with the field on at unchanged
strictness 0.50, as a control that the field does what it claims; (2) the
PSG-IPA sweep; (3) the MESA hold-out at the new point, against the `aasm15`
reconstruction only — the `oahi` variants encode a different rule and their
error directions are not interpretable here.

## Measured — and the plan does not survive first contact

The asymmetry is real and large, but it does **not** act on the profiles the
paper's headline rests on. PSG-IPA, five recordings, scorer-1 hypnogram,
`PSGSCORING_AROUSAL_DERIVATION=single`, all four cells measured:

| profile | detector | linearisation off → on | strictness 0.30 → 0.60 |
|---|---|---|---|
| `aasm_v3_rec` | rule cascade | **bias +1.77 → −1.43**, MAE 1.84 → 1.43, in-range 3/5 → 5/5, severity 4/5 → 5/5, hypopnoeas 208 → 116 | **no effect at all** (+1.77 both) |
| `aasm_v3_breath` | breath-graded | **no effect at all** (bias −0.29 both) | (effective by construction) |

Apnoea counts are unchanged throughout (316), as expected: linearisation
touches only the hypopnoea channel.

**Why the two knobs are disjoint.** `hypopnea_strictness` is read only inside
`score_hypopneas_breathwise`, i.e. on the graded branch. And that branch never
sees the envelope: `_run_breath_analysis` receives `hypop_flow` — the *raw*
signal — and runs its own `bandpass_flow`, computing breath amplitudes
directly. The linearisation lives in `_setup_hypop_channel`, which builds
`hypop_env` for the envelope cascade.

So the graded detector is non-linearised on **both** cohorts and therefore
internally consistent, while the rule cascade is non-linearised on one and
linearised on the other. The instruction to "enable linearisation and re-derive
strictness as one experiment" cannot be executed as written: on the profile
that owns strictness, linearisation has no effect; on the profile where
linearisation has a large effect, there is no strictness to re-derive.

**Consequence for the cohort asymmetry.** The linearisation asymmetry is *not*
the explanation for the opposite bias directions of `aasm_v3_breath` /
`aasm_v3_prob` (+0.17 on PSG-IPA against −13 to −14 on MESA), because those
profiles never linearise on either cohort. It remains a plausible contributor
for `aasm_v3_rec` and the other cascade profiles, and it is a genuine defect in
its own right: the same profile name preprocesses differently depending on the
montage.

**No default is changed.** Every shipped profile keeps
`hypopnea_force_linearisation = False`, golden 8/8 unchanged, 686 tests green.
Table 1 and Figure 1 of the paper therefore stand as published.

## A second confound, found while measuring

`aasm_v3_breath` gives bias **+0.17** under the default multi-derivation arousal
path and **−0.29** under `PSGSCORING_AROUSAL_DERIVATION=single` — a shift of
0.46/h from the arousal derivation alone, with nothing else changed. Any
reported bias must state which derivation produced it. The paper's Table 1 was
produced under the default (multi); the measurements in this entry under
`single`, as the work instruction requires for reproduction runs.

# v0.15.2 — 2026-08-08 — a rounding regression that a report showed and the tests could not

**Fix — published indices are back to one decimal.**

A clinical report printed `Arousal index (AI) 57.906 /u` next to
`Respiratoire arousal-index 16.8 /u`, and `ODI 3% 24.05 /u` where the same
recording read `24.0 /u` in early August.

Cause: v0.14.7 replaced twelve copies of `max(hours, 0.001)` with the shared
`per_hour()` helper. The helpers it replaced — `_safe()` in `arousal.py` and
`safe_r()` in `utils.py` — both defaulted to **one** decimal. Nine of the new
call sites in `arousal.py` were given an explicit `3`, and two in `spo2.py` a
`2`. The denominator fix was correct; the rounding rode along with it.

An arousal index is not knowable to 0.001/h. Where the boundary of an arousal
lies is a scorer's judgement with a spread of seconds, so three decimals assert
a precision the measurement does not have — in a clinical document that is
misleading, not merely untidy.

Affected: `arousal_index`, `nrem_arousal_index`, `rem_arousal_index` (both the
detection path and the post-LGBM recompute), the `severity` classification
input, `rera_index`, `rdi` in the arousal summary, and `odi_3pct` / `odi_4pct`.
Values change only in the digits after the first decimal.

**Not affected: the AHI family and the paper.** `respiratory.py` and the
golden-covered indices always used the default. Re-running the golden harness
after this change reports **0 changed values, 0 removed, 16 added**.

## Why the existing tests did not catch it

The golden digest rounds to one decimal itself (`_r(x, nd=1)` in
`test_golden_output.py`) before comparing. A change *from* one decimal *to* two
or three is invisible by construction: the digest discards exactly the
difference. And the arousal indices were not in the digest at all — only
`n_flat` and `n_nested` were.

Two guards, because the two failures are different:

- `arousal_index` and `rera_index` added to the golden digest — 16 new fields,
  no existing value touched. This guards the **value**.
- `tests/test_index_precision.py` — asserts the published indices carry one
  decimal, and that no `per_hour()` call site overrides the rounding at all. A
  deviation is a presentation decision and belongs in review, not as a third
  argument in a line about denominators. This guards the **presentation**.

# v0.15.1 — 2026-08-08 — a stage index needs enough of that stage, and an override should say so

## Added — the REM-AHI now says how much REM it rests on

At 22.5 minutes of REM a single event is already 2.7/h. That number is
mathematically correct and clinically unusable: it reads as a measurement while
being the rounding of a handful of events.

This is **not** the same as the denominator bugs fixed earlier this week. There
the index did not exist and was shown anyway; here it exists but cannot be
trusted. That difference decides the response: qualify rather than omit, so the
reader sees the number *and* what it rests on.

New in the respiratory summary: `rem_min`, `nrem_min`, `ahi_rem_reliable` and
`ahi_rem_caveat`. Purely additive — every existing number is unchanged and
golden is byte-identical.

The 30-minute threshold is not newly invented: `_compute_phenotypes` already
required it before the REM-predominant phenotype may be stated. One threshold
for one question, with a test that fails if the two drift apart.

## Added — active environment overrides are reported

`PSGSCORING_BREATH_CAND_MIN_DUR`, `PSGSCORING_BREATH_AROUSAL_LATENCY` and
`PSGSCORING_BREATH_TEMPLATE_DUR` override profile values. They exist to measure
without mutating profiles, but they mean the same profile name can behave
differently on two machines — and the provenance block, which exists to show
the *execution* rather than the choice, said nothing about it.

`meta.env_overrides` now carries whatever is active (empty in the normal case),
and the report prints a provenance row when it is not empty.

621 tests green, golden unchanged.

---

# v0.15.0 — 2026-08-07 — two names and one measure, each promising more than it delivered

## Changed — `aasm_v3_prob` no longer calls itself "fully probabilistic"

It was not. Only the **hypopnea** axis is graded. Apneas come from the same rule
cascade as every other profile, and measurement shows their confidence saturates
at the cap of whichever rule assigned it — on the golden cases, **0.95 for every
obstructive event and 0.90 for every central one**. Within a class the score
discriminates nothing.

Renamed to "graded arousal axis", with the limitation spelled out in the
description. The profile key is unchanged, so no configuration breaks. Grading
the apnea axis is detector work that has not been done; a test fails if an
apnea-detector choice ever appears without the names being revisited.

## Added — `hypoxic_burden_local_baseline` (default off)

The report was that hypoxic burden is *underestimated* on sustained hypoxemia.
Measured, that direction is wrong: a flat baseline of 85 % gives **exactly the
same** burden as 96 % at equal desaturation depth (21.57 both), which is correct
— Azarbarzin's measure is by definition the area under the patient's *own*
baseline.

The real defect is drift. `baseline = max(local, global)` puts the night-wide
95th percentile under every event, so events late in the night are measured
against a baseline from early in the night:

| recording | burden |
|---|---:|
| flat 96 % | 21.57 |
| flat 85 % | 21.57 |
| **drifting 94 % → 82 %** | **41.19** |

Nearly double, with identical events — the COPD and obesity-hypoventilation
picture. With the flag: **21.76**, back in line with the flat cases, while those
themselves do not move.

Default off, because this shifts a published quantity on every recording with
drift. The code had warned about exactly this in a v0.4.4 review, proposing to
gate the override below ~88 %; this flag is that gate, made explicit rather than
hidden in a threshold.

Golden unchanged, 524 tests green.

---

# v0.14.9 — 2026-08-07 — RERA de-duplication on interval overlap

## Fixed — the dedup test compared onsets, not intervals

RERA candidates were excluded as duplicates when their onset lay within 5 s of
an already-scored event. That test misses a candidate starting six seconds later
but falling entirely inside the event, and conversely calls two things the same
when they happen to start close together without touching at all.

Replaced by interval overlap — but not by *any* overlap, which measurement
showed to be too blunt:

| candidate type | overlap with a scored event |
|---|---|
| rejected hypopnea candidates | **0.83 – 1.00** |
| flattening sequences | **0.06 – 0.22** |

The first group are unmistakably the same episode, counted twice. The second are
long flow-limitation sequences clipping the edge of a hypopnea — flow limitation
*beside* that event, not a duplicate of it.

The rule is therefore **"more than half the candidate lies inside a scored
event"**. That threshold sits in the measured gap rather than in a hunch, and a
test asserts it stays between 0.22 and 0.83.

One golden line moves: `rdi: 56.8 → 37.8` on `arousal_autodetect_breath`, from
three rejected candidates at 0.83–1.00 overlap that the old exact-onset test let
through as separate RERAs. Every AHI, every event count and `mesa_shhs` are
unchanged — the published figures do not move.

510 tests green.

---

# v0.14.8 — 2026-08-07 — two indices that described a different recording than the report

## Fixed — positional AHI was computed before the events were final

`analyze_position` runs at step 6 on the event list as it stands there. Step 7b
— the breath-graded detector — then replaces **every hypopnea**. Without a
recompute the AHI-per-position still described the envelope detector while the
rest of the report showed the graded events.

This affects every profile with `hypopnea_detector="breath_graded"`:
`aasm_v3_breath`, `aasm_v3_prob`, `aasm_v3_breath_dual`, `aasm_v3_prob_dual`.
And through `ahi_per_pos` it reached the positional phenotype, so the verdict
*"candidate for positional therapy"* rested on the wrong event set.

The position analysis is now recomputed after the merge. Recomputed rather than
patched: the position distribution itself does not change, only the events
inside it, and `analyze_position` is the one place that couples the two.

## Fixed — "AHI excluding noise" could exceed the AHI

`ahi_excl_noise` counted over **all** events; `ahi_total` counts apneas +
hypopneas and deliberately excludes `uncertain` — an apnea the effort
classifier could not subtype. On a recording with such events the *filtered*
index came out higher than the unfiltered one: "AHI excl. noise 14.2 beside AHI
12.8", which looks impossible and makes a reader doubt both.

The denominator was never the problem — both use the same `total_sleep_h`. It
was the numerator, which now covers the same population, so the difference is
the noise filter and nothing else.

Golden unchanged, 510 tests green.

---

# v0.14.7 — 2026-08-07 — the same floor was in twelve places

## Fixed — `max(hours, 0.001)` survived in eleven more modules

v0.14.6 removed the denominator floor that turned an uncomputable index into
the event count × 1000. It removed it from **one** module. The same line was in
eleven others:

```
arousal.py   7×   pipeline.py  3×   plm.py  1×   spo2.py  1×
```

So on the recording that exposed this, the AHI was fixed while the **arousal
index, PLM index, ODI, RERA index and RDI were all still count × 1000**. The
report looked repaired because the headline number was.

All twelve now go through `psgscoring/indices.py`:

```python
def per_hour(n, hours, ndigits=1):
    if n is None or hours is None: return None
    if float(hours) <= 0:          return None
    return round(float(n) / float(hours), ndigits)
```

The RDI follows: with no RERA index there is no RDI, because AHI + nothing is
not a number.

A test walks every module in the package and fails on any
`max(… / 3600, 0.001)` that reappears. One rule in one place beats twelve
copies that were meant to stay in step and did not.

Golden unchanged, 499 tests green.

---

# v0.14.6 — 2026-08-07 — an index that could not be computed was reported as a number

## Fixed — a floor on the denominator turned "unknown" into count × 1000

A real report carried the headline **REI 81000.0/h → Severe SAS → CPAP therapy**
for 81 hypopneas. Every link in the chain was individually defensible:

1. the recording was a **polygraphy** — no EEG — but the analysis form
   *required* an EEG channel;
2. so the nasal pressure was entered to get past it;
3. YASA staged on that flow signal and produced a hypnogram;
4. the artefact detector looked at the same non-EEG channel and rejected
   **1078 of 1078 epochs** — entirely correctly;
5. no sleep time was left as a denominator;
6. `max(total_sleep_s / 3600, 0.001)` floored it at **3.6 seconds**, so every
   index became the event count × 1000 — and passed through the severity
   classifier untouched.

The floor is gone. `idx()` now returns `None` when the denominator is zero, and
the summary carries `indices_computable`, `index_denominator_h` and
`index_unavailable_reason`, which names which of the three causes it was: no
hypnogram, wake only, or everything masked as artefact.

**Returning 0 would not have been better.** "AHI 0.0" alongside 81 scored events
reads as *no events* — reassuringly wrong, which is clinically worse than
visibly wrong. An index that cannot be computed must be absent, not zero.

The same reasoning reaches the robustness grade: with no indices there is no
sweep width, so the grade is `None` rather than a width of 0 — which would have
produced **grade A ("robust, diagnosis stable")** on a recording where nothing
could be computed at all.

`ahi_rem` follows: no REM sleep now yields `None`, not 0. There is no REM to
express events per hour of.

Correct value for that recording, over recording time: **9.0/h — mild.**

Golden unchanged, 499 tests green. The floor was never exercised by any golden
case, which is exactly why it survived this long.

---

# v0.14.5 — 2026-08-06 — a gate that measured the wrong property

Every existing clinical, dataset and legacy profile is byte-identical: golden
unchanged, 488 tests green, and tests assert that no such profile moves on any
new axis.

## Added — a per-channel thermistor gate

`assess_flow_sensor_agreement` decides, on every recording, whether apneas are
scored on the thermistor or on the nasal pressure. It correlates the two
sensors' **envelopes**. That is not the same question as "does this thermistor
follow respiration", and the difference is not academic.

Six synthetic signals, all breathing at exactly 0.25 Hz — perfect respiratory
agreement by construction — differing only in their slow amplitude modulation:

| difference between the channels | agreement |
|---|---:|
| identical modulation *(what the existing unit test builds)* | 1.000 pass |
| thermistor delayed 1 s (thermal lag) | 1.000 pass |
| modulation shifted 90° | 0.002 **reject** |
| modulation at a different slow frequency | −0.038 **reject** |
| no modulation on the thermistor | 0.094 **reject** |
| modulation in antiphase | −0.985 **reject** |

The measure is decided entirely by amplitude-modulation covariation and is
blind to whether both sensors see the same breathing. Two physically different
transducers do not modulate their amplitude alike. The existing test could not
catch this: it builds both channels from the same `1 + 0.5·sin(2π·0.01·t)`
term, and that term *is* the envelope, so they correlate 1.000 by construction.

Measured on **9 distinct Somnomedics montages**: the gate rejects 8. Three of
those have a thermistor carrying **98 % of its power in the respiratory band**
and sharing its breathing frequency with the nasal pressure to within 0.002 Hz.

`assess_thermistor_band_power` asks the single-channel question instead: what
fraction of this channel's power falls in the respiratory band, as a median
over ten windows spread across the night. On the same 9 montages:

```
0.982  0.981  0.977  0.970   |   0.441  0.396  0.318  0.036  0.000
```

A gap of 0.53 with nothing in it. `THERMISTOR_BAND_POWER_MIN = 0.70` sits at
its midpoint — maximum margin to both classes — and reads as "at least 70 % of
this channel's power is respiration". Contrast the existing threshold, whose
own comment records that it separates nothing: *"bekend-slecht liep tot +0,225,
bekend-goed begon op +0,226 … bewust conservatief gekozen, niet afgeleid"*, and
that on 25 MESA recordings agreement correlates with dual confirmation at
r = +0.07.

Selected by `PostProcessingRules.thermistor_gate`, **default unchanged**
(`"envelope_agreement"`). Enabled only on `aasm_v3_breath_dual` and
`aasm_v3_prob_dual` — both new, both exploratory, so no existing outcome moves.
Without it those two are identical to their single-sensor parent on 8 of 9
montages: expensive no-ops.

⚠️ Still unmeasured: whether either gate produces AHIs closer to human scoring.
n = 9 is small and the threshold should be re-derived on more recordings.

## Fixed — a flat channel was given a fabricated agreement score

The guard tests `float(np.std(a)) == 0` exactly, but `filtfilt` on a constant
yields numerical noise of order 1e-15, so the standard deviation is not
precisely zero and the guard lets it through. A correlation is then computed
over that noise. On a real recording with an all-constant thermistor this
produced `agreement = 0.026` with the reason *"the thermistor does not follow
respiration like the nasal pressure"* — a fabricated number and a misleading
explanation, both reaching `thermistor_check` and the report.

The old path is left untouched (any change there moves the AHI on every
recording); the new measure uses a relative test and returns a true `0.000`.

---

## Also in this release — the empty square in the profile grid

The profiles varied along two independent axes and the crossing was empty:

| profile | hypopnea detector | apnea sensors |
|---|---|---|
| `aasm_v3_rec` | threshold | one |
| `aasm_v3_breath` | breath-graded | one |
| `aasm_v3_prob` | breath-graded + arousal weight 0.70 | one |
| `aasm_v3_dual` | threshold | two, additive |
| `aasm_v3_fusion` | threshold | two, agreement-weighted |
| **`aasm_v3_breath_dual`** | **breath-graded** | **two, additive** |
| **`aasm_v3_prob_dual`** | **breath-graded + 0.70** | **two, additive** |

Both **`exploratory`**, both **off unless selected**, and both derived from
their parent with `dataclasses.replace()` rather than copied — so a future
change to `aasm_v3_breath` reaches its dual variant automatically. Every
existing profile is byte-identical: golden unchanged, 471 tests green.

## The two axes really are independent — but not for the reason expected

The design note warned that `flow_reference` had to be `"hypopnea"` because
otherwise the breath segmentation would run on the thermistor: a slow, rounded
signal, measuring something other than what `aasm_v3_breath` was validated on.

**That premise is wrong.** `respiratory.py` calls `_run_breath_analysis(hypop_flow
if hypop_flow is not None else flow_data, …)` — breath segmentation always runs
on the hypopnea channel, whatever `flow_reference` says. The field cannot move it.

`flow_reference="hypopnea"` is still correct on both profiles, for a different
reason: **the arousal analysis reads `ref_flow`** (`pipeline.py:703`), and its
arousals are one half of the noisy-OR confirmation inside the graded detector.
Under an additive thermistor the apnea channel points at the thermistor even
when it fails the quality gate — the second detection pass makes that harmless
before the apnea count, but the arousal coupling knows nothing of that pass.
So the path exists; it just runs through the arousals rather than the breaths.

Three further checks, all reading the code rather than reasoning about it:

- **The graded detector never touches apneas.** Step 7b keeps every non-hypopnea
  event from the envelope detector and merges them back untouched
  (`pipeline.py:731`). The second sensor operates on exactly the list the graded
  detector leaves alone.
- **De-duplication is on timestamps, not indices.** `corroborate_apnea_events`
  matches on IoU ≥ 0.20 over `onset_s`/`duration_s`, so the breath time raster
  cannot disturb it.
- **`sensor_agreement` is set before step 7b** and only on apneas, so it behaves
  identically under either hypopnea detector — a future `breath_fusion` is
  therefore mechanically possible.

One genuine coupling does exist and is intentional: the dual-merged apnea list
becomes `exclude_intervals` for the graded detector, so an added apnea can
suppress a hypopnea in the same window. That is the correct direction — one
window is one event — but it means the *hypopnea* count is not monotone in the
sensor axis. The *apnea* count is, and that is what the invariant test asserts.

## What is and is not measured

Measured, in `tests/test_dual_combinations.py`: on a single-flow montage each
child is **event-for-event identical** to its parent, exactly as `rec`, `dual`
and `pressure` coincide on PSG-IPA. All PSG-IPA numbers published for
`aasm_v3_breath` and `aasm_v3_prob` therefore carry over unchanged. On a
two-sensor montage the second sensor adds apneas and removes none.

⚠️ **Not measured: whether the sensor axis improves agreement with human
scoring.** PSG-IPA has one flow channel and no thermistor, so this axis is not
observable there — the same limitation `aasm_v3_fusion` already carries. That
needs MESA or a cohort with two working flow sensors. Until then these are
research instruments, and `aasm_v3_prob_dual` stacks two unvalidated axes at
once, putting it furthest from any measured operating point.

The prior decision gate — does dual-sensor detection differ enough from
single-sensor on our own montages to be worth having — is **still open**. The
existing evidence is indirect: on MESA only 19 % of 1785 apnea detections appear
on both sensors, and on four consecutive AZORG SOMNO recordings the thermistor
contributed **zero** apneas while `Flow Th.` was in the montage.

---

# v0.14.4 — 2026-08-05 — two thresholds hiding inside graded models

Two new profiles, both **`exploratory`** and both **off unless selected**. Every
existing profile is byte-identical: golden unchanged, 451 tests green, and tests
assert that no existing profile moves on any of the new axes.

## Added — `aasm_v3_prob`: the arousal criterion was still a threshold

The breath-graded detector evaluates every AASM criterion on a sliding scale —
except one. Confirmation is a noisy-OR,

```
p_confirm = 1 − (1 − p_desat)(1 − p_arousal)
```

and `p_arousal` **jumped** to `arousal_weight = 0.90` the moment an arousal fell
in the coupling window. So `p_confirm ≥ 0.90` however small the desaturation: a
threshold sitting in the middle of an otherwise continuous model, and a free
pass for any candidate with an arousal nearby.

This was found by measurement, not by inspection. On PSG-IPA SN1
`aasm_v3_breath` marks ten events that **not one of the twelve scorers** marked;
five have a desaturation below 1.2 % while the lowest consensus event sits at
2.16 %. Desaturation separates those two groups with **AUC 0.818**, the event
confidence with only 0.684 — the axis making the error is the one that does not
count, and the probability the whole design rests on separates weakly.

`aasm_v3_prob` is `aasm_v3_breath` with `hypopnea_arousal_weight = 0.70` and
`hypopnea_arousal_latency_grading = True`. An arousal becomes an argument rather
than a proof: without desaturation a candidate now needs `p_flow · p_dur > 0.71`
instead of 0.56, while an arousal *with* partial desaturation still clears the
operating point through the noisy-OR.

Measured on PSG-IPA (n = 5, manual hypnogram, so no staging effect):

| configuration | \|bias\| | F1 med | consensus recall | precision | invented |
|---|---|---|---|---|---|
| `aasm_v3_rec` | 1.84 | 0.343 | 0.588 | 0.577 | 97 |
| `aasm_v3_breath` | **0.29** | 0.434 | 0.500 | 0.722 | 69 |
| w=0.90 + latency | 0.42 | 0.441 | 0.500 | 0.765 | 60 |
| **w=0.70 + latency** | 0.89 | **0.453** | 0.500 | 0.788 | 47 |
| w=0.55 + latency | 1.47 | 0.450 | 0.470 | 0.862 | 37 |
| w=0.40 + latency | 2.01 | 0.441 | 0.439 | 0.885 | 29 |

*"Invented" = events overlapping no human event from any of the twelve scorers.*

Events nobody marked fall monotonically 69 → 26 as the weight drops, which
confirms they came in through this branch. Below 0.70 it starts costing
consensus events, so 0.70 is the lowest weight that is still free. Latency
grading alone is a gain at no cost, independently confirming the v0.13.0
measurement.

Note that AHI bias and event agreement do not peak together: `aasm_v3_breath`
keeps the best bias because its operating point was chosen as the zero-bias
point. Re-deriving `hypopnea_strictness` at weight 0.70 is the open next step.

## Added — `aasm_v3_fusion`: sensor agreement as a weight, not a gate

`assess_flow_sensor_agreement` computes a continuous envelope correlation
between thermistor and nasal pressure — 0.32 to 0.71 on real recordings — and
that number was reduced to `usable` yes/no. Above the line the sensor counts
fully, below it not at all. The same shape of mistake as the arousal axis, one
level up.

`aasm_v3_fusion` is `aasm_v3_dual` with `thermistor_agreement_weighting = True`.
Every apnea carries `sensor_agreement`, and an apnea whose only support is the
thermistor has its confidence scaled by that value — so a detection from a
sensor agreeing at 0.32 enters at a third of its confidence instead of either
fully or not at all. Nothing is rejected; the thermistor stays additive,
because falsifying with a sensor you do not trust is how one real recording lost
83 central apneas.

⚠️ **Unvalidated, and not measurable on PSG-IPA.** That montage has a single
flow channel and no thermistor — which is exactly why `aasm_v3_rec`,
`aasm_v3_dual` and `aasm_v3_pressure` are identical there to the decimal.
Testing this axis needs MESA or a cohort with two flow sensors.

## Added — `hypopnea_desat_width`, and the measurement that says not to touch it

Exposed as a profile field for completeness, default 0.80 = unchanged. Widening
it was the obvious response to the missed events, and it does not work — for a
reason worth recording.

All five missed consensus events on SN1 **were candidates**; none was invisible.
They were rejected on probability: p = 0.245–0.464 against a bar of 0.50, with
desaturations of 2.15–3.15 %. From p = 0.245 at p_desat = 0.257 it follows that
reduction × duration × template ≈ 0.95 — the detector found the flow reduction
and duration entirely convincing and dropped the event on a 2.2 % desaturation,
one that twelve of twelve scorers marked.

But `graded()` is a sigmoid **centred on the threshold**, so a value below 3 %
can never exceed p_desat 0.5 at any width. The ceiling for those five is
p = 0.42–0.48. Measured over the cohort, widening loses more than it gains:

| desat width | F1 med | consensus recall | precision |
|---|---|---|---|
| 0.4 | **0.456** | 0.500 | **0.818** |
| 0.8 (default) | 0.453 | 0.500 | 0.788 |
| 1.2 | 0.438 | 0.455 | 0.803 |
| 2.4 | 0.392 | 0.394 | 0.800 |

Narrowing beats widening: making that axis *more* decisive improves both F1 and
precision. The knob the SN1 analysis actually points at is the sigmoid's centre
— the 3 % threshold itself — and that is a rule change, not a calibration.

# v0.14.3 — 2026-08-04 — the saturation channel was being read as EEG

Golden byte-identical; no scored event moves.

## Fixed — `SpO2` could be assigned to the `eeg` role

`"o2"` is one of the `eeg` patterns and a substring of `"SpO2"` and `"SaO2"`.
On a montage without an EEG channel the role therefore claimed the saturation
trace, and `_pick_eeg()` reads that role directly — the arousal analysis would
have run on the SpO2 curve.

Same class of trap as `"pr"` → `"Pres"` in v0.14.2, and fixed the same way: the
`eeg` role no longer accepts a channel already claimed by another role. Blocked,
`_pick_eeg()` falls back to its own stricter list (EEG/C3/C4/F3/F4/CZ, without
O1/O2) or to nothing — both better than a saturation curve read as EEG.

The exclusions are declared in one place (`_ROLE_MAY_NOT_TAKE` in `utils.py`)
rather than special-cased per role. Deliberately not a general "one channel, one
role" rule: on a single-flow-channel montage, `flow` and `flow_thermistor`
correctly point at the same channel.

# v0.14.2 — 2026-08-04 — the summary now describes the events the report shows

One intended output change, in `oahi` and the apnea-type counts of the
**selectable** profiles (issue #18, below). Everything else stays
**byte-identical**, and no scored event moves anywhere: `ahi_total` and the
event count are unchanged in every case.

Golden re-blessed, and the whole diff is two lines in one case:

```
poor_quality  .resp.oahi_all    6.3 -> 0.0
              .resp.oahi_conf60 6.3 -> 0.0
```

Both obstructive apneas in that case are CSR-reclassified to central, so the
obstructive index is genuinely zero once you count the events the report
displays. The other seven cases are unchanged, `mesa_shhs` and `chicago_1999`
are pinned, and the PSG-IPA reproducibility suite asserts on AHI, which this
cannot move — it was not re-run here because the dataset is not configured on
this machine.

## Changed — `summary_after_reclassification` is ON for selectable profiles (issue #18)

The last `_compute_summary()` runs at step 9; step 11 then reclassifies events
(CSR-driven obstructive→central, mixed decomposition) and replaces the event
list. `n_obstructive`, `n_central`, `n_mixed`, the type indices and `oahi`
therefore describe the state *before* that reclassification, while the event
list, the confidence table and every per-event figure describe the state after.

In the report this shows up as a gap between the `n` column and the confidence
breakdown, exactly equal to `n_csr_reclassified`. It is not cosmetic: `oahi`
moved 32.2 → 28.6 on one real recording, across the severe/moderate boundary.
`ahi_total` is unaffected — the reclassification relabels events, it neither
adds nor drops any.

The new flag recomputes the summary after step 11. It is **on for every profile
a clinician can select** — the clinical family plus the two v3 arms in the
exploratory family, which sit in the same dropdown — and **off for the
reproduction profiles** `mesa_shhs` and `chicago_1999`, which must stay
byte-identical to the published numbers.

The dataclass default stays `False` on purpose: forgetting to switch it on for a
new clinical profile leaves the old behaviour, which is visible and harmless,
whereas forgetting to pin a new dataset profile would silently break NSRR
reproduction. The safe failure mode is the default.

The recompute *merges* rather than replaces, so the RERA/RDI/REM-NREM keys that
steps 9b/9c add survive — the same trap that was fixed in v0.7.4.

Note that the robustness sweep calls `detect_respiratory_events` directly rather
than the pipeline, so the strict/sensitive arms of `profile_comparison` never
read this flag; only a run where such a profile is the primary one is affected.

## Fixed — `PLMl`/`PLMr` were never auto-detected

`leg_l`/`leg_r` carried `"plm l"` and `"plm-l"`, with a separator. Matching is
substring-based, so `"plml"` matched neither and SOMNOmedics leg channels went
undetected; the PLM analysis silently depended on a manual channel choice in the
UI. Added `"plml"`/`"plmr"` — deliberately *not* a bare `"plm"`, which would
match both channels and, since each role takes the first match in EDF order,
assign one channel to both legs. `detect_channels()` now also refuses to return
the same channel for both legs.

## Fixed — the nasal pressure could be used as the heart-rate channel

The `pulse` role's shortest pattern is `"pr"`, a substring of `"Pres"` and
`"Pressure Flow"`. On a montage without its own pulse channel the role claimed
the nasal pressure, after which `analyze_heart_rate` treated a flow waveform as
bpm. `pulse` no longer accepts a channel already taken by a flow role.

## Added — heart-rate plausibility is now explicit

Three recordings reported a minimum heart rate of 20.2 · 32.6 · 20.0 bpm. Two of
those sit exactly on the lower bound of the `[20, 250]` plausibility filter —
the signature of an oximeter briefly letting go, not of bradycardia. The
absolute extremes are as trustworthy as the single worst sample.

`analyze_heart_rate` keeps every existing key and adds `hr_p1`, `hr_p99`,
`hr_implausible_pct`, `hr_source`, `hr_reliable` and `hr_unreliable_reason`. No
silent correction: the numbers stay, the verdict is stated, and the report can
choose what to show. Without a pulse channel the pipeline still falls back to
the raw ECG — a waveform, not bpm, with no R-peak detection on the way — and
that path is now marked unreliable rather than reported as a heart rate.
`hr_data` itself is unchanged, so the arousal analysis reads exactly what it
read before.

## Fixed — a failed thermistor check dropped the channel without a trace

When `assess_flow_sensor_agreement` raised, the exception path removed the
thermistor without setting `flow_thermistor_rejected`, so the report said
"thermistor not in montage" for a channel that was in the EDF and had merely
failed to be assessed. The rejection and its reason are now recorded, exactly as
in the normal rejection path. Metadata only.

# v0.14.1 — 2026-08-03 — two report figures that were reading the wrong thing

Both found by comparing two reports on one real recording — `aasm_v3_rec` on
v0.13.1 against `aasm_v3_dual` on v0.14.0. The headline was stable across both
runs (severe CSAS, mild OSA, same automated conclusion); these are the numbers
underneath it. Every existing profile stays **byte-identical**; the golden
harness confirms it.

## Fixed — "Rule 1B / CMS (≥4% desat)" was not a 4% figure

`_compute_dual_ahi` read the **`strict` arm of the robustness sweep**. But
`aasm_v3_strict` is a *conservative variant of Rule 1A*: it keeps
`desat_or_arousal` with `desat_threshold = 0.03` and differs in the stability
filter (CV 0.45 → 0.30), breath-level detection and nadir window. No ≥4%
criterion was ever applied to the number printed under a "≥4% desat" heading —
in **any** release since the field was added in v0.11.0.

Rule 1B now comes from the final event list: every apnea, plus the hypopneas
whose linked desaturation reaches 4%. Filtering the Rule 1A list is *exact*
rather than approximate — the two rules share the flow-reduction and duration
criteria, and a 4% desaturation is by definition also a 3% one, so every Rule
1B hypopnea is already in the Rule 1A list. What drops out is precisely the
hypopneas that qualified on arousal alone or on a 3–4% desaturation.

Consequences:

- The apnea set matches `ahi_total` (`uncertain` excluded on both sides), so
  **Rule 1B can no longer fall below the apnea index** — it did, reading
  10.3/h on a night with 78 apneas, beside Rule 1A at 33.9/h.
- With no successful SpO₂ analysis the row is **omitted** rather than
  degrading to "apneas only", which looks like a result.
- For a profile that already requires ≥4% (`cms_medicare`, `aasm_v1_rec`) its
  own headline *is* Rule 1B; Rule 1A still comes from the sweep's `standard`
  arm.

## Fixed — five analyses read the apnea channel when they wanted flow

Apnea detection is not the only consumer of a flow signal. The AHI-interval
sweep, baseline anchoring, the arousal/RERA coupling, Cheyne-Stokes detection
and the ventilatory burden all took one — and all took the **apnea** channel.

Under a substitutive profile that is harmless: the apnea channel *is* the
recording's flow. Under `aasm_v3_dual` it is not. That profile deliberately
keeps the thermistor as the nominal apnea channel even when it fails the
quality check, because the second detection pass makes a bad channel unable to
*remove* events. These five got no second pass, so they inherited exactly the
substitutive failure the additive design exists to avoid.

New profile field **`flow_reference`**:

- `"apnea"` — the apnea channel. **Default; current behaviour.**
- `"hypopnea"` — the hypopnea channel: nasal pressure where the montage has
  one, the single available channel otherwise. Set on `aasm_v3_dual`.

`"hypopnea"` is also the AASM-correct answer on its own terms — the manual
assigns quantitative flow assessment to nasal pressure and treats the oronasal
thermistor as qualitative, able to show the absence of flow but not to grade
it. `meta["flow_channels"]["reference_sensor"]` reports which channel was used.

Also fixed alongside it: the shared preprocessing cache (envelope, dynamic
baseline) is built by the primary pass on the apnea channel and was handed to
the sweep unconditionally. Once the sweep reads a different channel that cache
is stale — the sweep gets a fresh one, and only then.

On a synthetic montage whose thermistor carries noise, the ventilatory burden
reads **0.0% before and 59.9% after**, and the sweep's three arms stop
diverging between the two profiles.

## New — profile `aasm_v3_pressure`

The reference correction as a selectable algorithm of its own, so it can be
compared against the others on a real recording instead of only existing inside
`aasm_v3_dual`.

`aasm_v3_rec` in every respect — apneas single-sensor, on the thermistor where
the montage has a usable one — except that the five derived analyses read the
nasal pressure. One variable moved, nothing else.

| profile | apneas | derived analyses |
|---|---|---|
| `aasm_v3_rec` | thermistor¹ | thermistor¹ |
| `aasm_v3_pressure` | thermistor¹ | **nasal pressure** |
| `aasm_v3_dual` | **both, merged** | nasal pressure |

¹ where it passes the quality check; nasal pressure otherwise.

It is therefore **identical to `aasm_v3_rec` on any montage without a usable
thermistor** — which is most clinical montages, and all of the Somnomedics
recordings this was found on. It differs where a thermistor does pass, MESA/NSRR
among them.

# v0.14.0 — 2026-08-03 — dual-sensor apneas as a profile, after v0.13.2 was rolled back

v0.13.2 made the thermistor the apnea sensor wherever one was detected. On a
real recording with human scoring that took apneas from **93 to 0**, AHI from
26.2 to 8.6, and the conclusion from *moderate CSAS* — confirmed by the human
scorer — to *mild SAS*. 100 MESA recordings and 5 PSG-IPA recordings did not
show it; one recording with a human reference did. v0.13.2 was rolled back in
production the same day.

The lesson is narrow and worth stating plainly: **the AASM-specified sensor is
only the right sensor if it carries a usable signal.** On that montage the
thermistor's breathing envelope correlated with nasal pressure at ≈ 0.

## New — profile `aasm_v3_dual`

Apneas are detected on **both** flow sensors and merged, instead of choosing
one. The nasal pressure misses mouth breathing; the thermistor is too slow for
short events. Neither channel can veto the other.

**Every existing profile is byte-identical to v0.13.2** — the golden harness
confirms it — and `aasm_v3_rec` remains the default. This is opt-in.

- `postprocess.corroborate_apnea_events()` merges the two detections into
  `both` / `thermistor_only` / `pressure_only`; on overlap the thermistor
  defines the boundaries. Its default is **keep everything**: a detection
  seen by one sensor only is still an event. Corroboration can be *licensed*
  to discard single-sensor events, but nothing in this release licenses it.
- Measured before choosing that default: across 1785 apnea detections on
  MESA, only **19%** are seen by both sensors. Discarding the rest would
  remove four out of five events.

## New — `signal_quality.assess_flow_sensor_agreement()`

Envelope correlation between the two flow channels, reported as
`meta["flow_channels"]["thermistor_check"]`. It answers "does this thermistor
follow the breathing at all".

It gates the **substitutive** use only — where the thermistor *replaces*
nasal pressure for apneas, a bad channel deletes events, which is what
v0.13.2 did. Under `aasm_v3_dual` the second sensor is *additive* and can only
ever contribute events, so the gate does not block there; it still reports.

Honest limit: `THERMISTOR_AGREEMENT_MIN = 0.40` is **chosen, not derived**.
On MESA the metric does not predict corroboration (r = +0.07; the both-sensor
fraction is 0.13 above and below the threshold alike). It separates
"breathing" from "noise", not "good" from "better".

## Fixed — the apnea channel's baseline leaked into hypopnea scoring

When both roles resolved to the same channel, `respiratory.py` reused the
precomputed baseline for the hypopnea channel by comparing length and sample
rate. Any two channels in one EDF match on both, so the apnea channel's
baseline was reused for a *different* signal. The caller now passes channel
identity explicitly.

## Also

- `meta["dual_sensor_fallback"]` records that dual scoring was requested but
  not performed, so a silently single-sensor run is visible downstream.
- Auto-detected arousals now feed the respiratory rules only where the profile
  opts in (`arousal_limb_wired`); externally supplied arousals are always
  honoured, as before.

## Not fixed

Nasal pressure still is not √-linearised. Applying it makes agreement worse
(bias +1.77 → −3.15, MAE 1.84 → 3.15) because twelve bias corrections are
calibrated against the un-linearised behaviour. Re-tuning them was attempted
and failed. This is a known, documented gap, not an oversight.

# v0.13.2 — 2026-08-02 — the AASM two-sensor rule now actually fires

The dual-sensor machinery has been in place since v0.8: `_resolve_flow_channels()`
assigns **apnea to the oronasal thermistor and hypopnea to nasal pressure**, and
`_setup_hypop_channel()` builds a separate envelope and baseline for the second
channel. It was simply never fed on some montages.

## Fixed — the Somnomedics thermistor was invisible

Somnomedics/SOMNO names the thermistor `Flow Th.`. None of the
`flow_thermistor` patterns matched it — `"th."` is too short for `"therm"` —
so `flow_thermistor` stayed empty and the pipeline silently fell back to
scoring **apneas on nasal pressure**, with the thermistor sitting unused in
the EDF. Patterns `"flow th"` and `"thermal"` added.

Nasal pressure is the more sensitive sensor: it drops readily on partial
obstruction and on mouth breathing. The AASM specifies the thermistor for
apneas precisely to avoid that over-detection, so the silent fallback biases
apnea counts upward.

Detection is per-role and independent, so this affects nothing else:

| montage | nasal pressure | thermistor |
|---|---|---|
| Somnomedics | `Pressure Flow` | **`Flow Th.`** (was none) |
| MESA / NSRR | `Pres` | `Therm` (unchanged) |
| PSG-IPA | — | — (unchanged) |

PSG-IPA has no thermistor, so golden output and paper reproduction are
byte-identical. A test asserts `"flow th"` cannot claim `Pressure Flow` —
that would swap the roles.

## Fixed — `dual_sensor` meant "a hypopnea channel exists", not "two sensors"

`respiratory.py` set the flag to `True` as soon as any hypopnea channel was
present, without comparing it to the apnea channel. Since
`_resolve_flow_channels()` falls back to the *same* channel when the
thermistor is missing, the flag was on almost always. Its only consumer is
YASAFlaskified's report line *"apnea on thermistor, hypopnea on nasal
pressure (AASM)"* — which therefore asserted a methodology that had not been
applied, including on single-channel recordings.

`meta["flow_channels"]["dual_sensor"]` was already computed correctly (both
channels present); the pipeline now makes that the source of truth.

## Note on the fallback

The fallback itself stays: many montages carry a single flow channel, and
scoring on the available sensor beats refusing to score. What changes is that
it is no longer silent — consumers can read
`meta["flow_channels"]` to see which sensor was used for what, and
`dual_sensor` now answers the question it appears to answer.

# v0.13.1 — 2026-08-02 — audit of the breath detector: corrections, no scoring change

Every existing profile is **byte-identical** to v0.13.0, `aasm_v3_breath`
included; the golden harness confirms it. What changes is what the code
*claims*, what each event *carries*, and which knobs are *visible*.

## Fixed — `p_scored` was documented as something it is not

v0.13.0 described `p_scored` as "what fraction of scorers would mark this".
That claim was never tested. It is wrong.

PSG-IPA makes the target directly measurable — 12 scorers per recording, so
the fraction marking any given event is a countable number. Over the 163
events the detector scores:

| | |
|---|---|
| `p_scored` median | 0.693 |
| actual scorer fraction | **0.167** |
| correlation | **r = 0.194** |
| systematic offset | **+0.333** |

The *ordering* holds weakly — band 0.50–0.60 corresponds to an actual
fraction of 0.32, band 0.90+ to 0.58 — but the *level* is over 30 points too
high. An event presented as "★★★ ≥0.85" is marked by 44% of scorers, not 85%.

`p_scored` keeps its name (renaming would break consumers) and is now
documented as what it is: a ranking by rule conformity, not a probability.
Calibration to a real probability is achievable — PSG-IPA supplies the target
per event — but needs a fitted mapping on more than five recordings.

## Fixed — two silent omissions in the event record

- **`min_spo2` was hard-coded to `None`** while `_desat_at()` had already
  computed the nadir. It is now reported.
- **`hypopnea_central` / `hypopnea_mixed` disappeared**, because the breath
  detector does no effort-based subtyping. Step 7b now inherits the label
  from the overlapping envelope event, which `classify_apnea_type()` had
  already classified on the same window using effort, ECG and flattening.
  Recomputing it would require envelopes that exist only inside
  `respiratory.py`. Coverage is visible as `n_subtyped` /
  `n_envelope_subtyped`; where nothing overlaps the event stays `hypopnea`.

## Fixed — one real design fault

`candidate_floor` and `min_duration_s` were used in **both** passes: they
decide which breaths stay out of the baseline *and* which become candidates.
Raising the floor lowers the baseline (less excluded) and raises the gate —
two opposing effects in one knob, which showed up as non-monotonic behaviour
(0.10 gave SN1 +10% but SN4 −14%). Pass A now has its own `exclusion_floor` /
`exclusion_min_duration_s`, defaulting to the candidate values.

## Changed — "strictness is one calibrated axis" was not true

A sensitivity analysis on PSG-IPA SN1 and SN4 refutes the assumption under
which the hand-picked constants were left unexamined. **None is inert:**

| parameter | SN1 | SN4 |
|---|---|---|
| `flow_width` | +6% … −39% | +27% … **−68%** |
| `recovery_margin` | −6% … **+42%** | −27% … **+68%** |
| `dur_width` | +10% … −39% | +14% … −50% |
| `stability_cv` | +3% … −35% | +23% … −36% |
| `use_template=False` | +10% | **+36%** |

There are six axes of comparable influence, of which one — `strictness` — is
calibrated. The module docstring now carries this table instead of the
promise. Eight values that were hard-coded inside the function are now
parameters at exactly their previous values: `sure_depth`, `default_lag_s`,
`arousal_weight`, `template_floor`, `template_center_frac`,
`template_width`, `n_largest`, `min_baseline_breaths`.

## Added — three scoring changes, all off by default

Each alters which events are scored and therefore leaves the calibrated
operating point, so each ships disabled. Measured on PSG-IPA:

| configuration | F1 med | F1 mean | pct | bias | MAE | severity |
|---|---|---|---|---|---|---|
| default | 0.434 | 0.512 | p17 | +0.17 | **0.29** | 4/5 |
| `candidate_min_duration_s=8` | 0.434 | 0.512 | p17 | +0.17 | 0.29 | 4/5 |
| `arousal_latency_grading` | **0.440** | **0.521** | p17 | −0.19 | 0.42 | **5/5** |
| `template_use_duration` | 0.416 | 0.505 | p15 | +0.25 | 0.37 | 4/5 |
| all three | 0.419 | 0.515 | p15 | −0.15 | 0.50 | 4/5 |

- `candidate_min_duration_s` changes **nothing** — identical on every measure
  and every recording. The asymmetry it addresses is real (the flow floor sits
  below the AASM threshold so marginal reductions reach grading; duration had
  a hard 10 s gate *and* a graded penalty around 10 s) but breaths last ~4 s,
  so runs quantise to 8/12/16 s and an 8 s run gets `p_dur ≈ 0.27` and never
  survives `strictness` 0.50.
- `template_use_duration` makes it **worse**. The template did use only
  `depth` of the three things it stored; filling that gap does not help.
- `arousal_latency_grading` is the only gain: four of five recordings
  improve, none worsens, severity 4/5 → 5/5. But MAE moves the wrong way and
  n = 5, so it stays off pending held-out confirmation.

The template now also computes the cluster periodicity the design called for
(`template["cycle_s"]`), as diagnostics only — letting it steer the score
would add a seventh unvalidated axis.

`_breath_env_overrides()` allows measuring these without mutating profiles,
following the `PSGSCORING_BASELINE_MODE` convention:
`PSGSCORING_BREATH_CAND_MIN_DUR`, `_AROUSAL_LATENCY`, `_TEMPLATE_DUR`.

---

# v0.13.0 — 2026-08-01 — breath-graded hypopnea detector (opt-in)

## Read this first if you reproduce published numbers

**`mesa_shhs` output changes in this release, and it is the channel-detection
fix that does it — not the new detector.**

| mesa-sleep-2408 | v0.12.2 | v0.13.0 |
|---|---|---|
| AHI | 30.4 | **22.0** |
| events | 291 | 234 |

Cause: `Pres`, the NSRR/MESA nasal-pressure channel, was claimed by the
`pulse` role, leaving `flow_pressure` empty so hypopneas were scored from the
thermistor. v0.13.0 assigns the sensors correctly (see v0.12.3 below). This
was verified by attribution: running v0.13.0 code with the *old* channel
patterns reproduces v0.12.2 exactly — AHI 30.4, RDI 30.4, 291 events.

Everything else in this release is byte-identical for existing profiles.
**To reproduce previously published MESA/NSRR numbers, pin
`psgscoring==0.12.2`.**

## Added — `aasm_v3_breath`, a breath-graded hypopnea detector

Registered as a **selectable, non-default** profile. `run_pneumo_analysis()`
still defaults to `aasm_v3_rec`, the legacy aliases still resolve to the
historical profiles, and YASAFlaskified — which passes
`scoring_profile="standard"` explicitly — is unaffected until its pin is
changed deliberately. It appears in that application's profile menu
automatically, because the menu enumerates this registry.

Why it is not the default despite winning on every aggregate measure:
PSG-IPA and MESA disagree on the *absolute* AHI level by ~16 points, and
until that is explained the existing clinical output stays in place. See
`docs/interim_conclusie_klinisch_gebruik.md`.

## Fixed — the AASM Rule 1A arousal limb reaches its consumers again

Issue #16: `run_arousal_respiratory_analysis()` returns its events nested,
while every scoring step read a flat key the auto-detection path never
populated. Repairing it restores arousals to YASAFlaskified's EDF+ export and
event API.

**Which profiles *act* on the repaired list is a profile choice with existing
behaviour as the default** (`PostProcessingRules.arousal_limb_wired`, default
False; only `aasm_v3_breath` opts in). Letting the existing profiles react
would change Rule 1B reinstatement, FRI-RERA and the LightGBM features at
once, none of which has been re-validated. Caller-supplied arousals
(`arousal_events=`) were never affected by issue #16 and are always honoured.

`PostProcessingRules.ml_arousal_features` (default False) covers the related
train/serve mismatch: the booster's top features are arousal-based but were
always zero at inference, so it keeps receiving zeros until retrained.

### Held-out confirmation on MESA/NSRR

50 records, seed 20260801, drawn from the 2055 with both an EDF and an NSRR
annotation. **Nothing was re-tuned**: `hypopnea_strictness` stayed at the
0.50 chosen on PSG-IPA. Both profiles ran on the same record with the NSRR
hypnogram, so only respiratory scoring differs. `scripts/validate_mesa.py`.

The reference is reconstructed, not counted — see that script's docstring.
NSRR's `Hypopnea`/`Unsure` labels carry no desaturation requirement; it is
applied afterwards by linking to the separate `SpO2 desaturation` events.
Reconstruction validated against published `oahi35`/`oahi45`: bias +0.11 /
+0.19 AHI, MAE 0.84 / 0.66, **r = 0.998** over 60 records.

| ref | profile | F1 med | F1 mean | prec | recall | bias | r | LoA | severity |
|---|---|---|---|---|---|---|---|---|---|
| oahi4 | `aasm_v3_rec` | 0.379 | 0.361 | 0.332 | 0.484 | −2.24 | 0.731 | −17.3…+12.8 | 29/50 |
| oahi4 | `aasm_v3_breath` | **0.442** | **0.411** | **0.365** | **0.571** | **−0.36** | **0.754** | **−14.8…+14.1** | 25/50 |
| oahi3 | `aasm_v3_rec` | 0.414 | 0.399 | 0.471 | 0.404 | −7.95 | 0.751 | −27.9…+12.0 | 23/50 |
| oahi3 | `aasm_v3_breath` | **0.464** | **0.432** | **0.507** | **0.494** | **−6.07** | **0.784** | **−24.8…+12.7** | 22/50 |

Paired per record: **31 better / 17 worse** on oahi4 (Wilcoxon
**p = 0.0069**) and **31 better / 18 worse** on oahi3 (**p = 0.039**). The
PSG-IPA finding replicates on data that had no hand in choosing the
operating point.

**Superseded in part — read the second round below before using this
table.** The two references above (`oahi35`/`oahi45`) credit no hypopnea that
qualifies on arousal alone, while Rule 1A does. That made every arousal-only
event a false positive and depressed both precision and recall. A third
reference reconstructed from `nsrr_ahi_hp3r_aasm15` — literally the rule
these profiles implement — changes the conclusion about severity.

**The apparent trade-off.** Event-level agreement improves and so do
correlation and limits of agreement, but severity concordance against
MESA's 4% definition falls from 29/50 to 25/50, and the loss is almost
entirely one cell:

| error | `aasm_v3_rec` | `aasm_v3_breath` |
|---|---|---|
| **Normal → Mild** | 7 | **10** |
| Moderate → Mild | 3 | 5 |
| Severe → Moderate | 2 | 5 |
| Mild → Moderate | 5 | 3 |

Higher recall (0.484 → 0.571) tips three more normal recordings over
AHI 5 — clinically the least comfortable direction, a false-positive OSA
call. And the gain is distributed as the mirror image of that risk:

| reference severity | median ΔF1 (oahi4) | better |
|---|---|---|
| Normal (n=16) | −0.002 | 6/16 |
| Mild (n=19) | +0.033 | 12/19 |
| Moderate (n=10) | **+0.096** | **9/10** |
| Severe (n=5) | **+0.153** | 4/5 |

The detector wins where there is disease and is neutral-to-harmful in
normals. MESA is a community cohort with many normal recordings; a sleep-lab
population is enriched for disease, which shifts the balance — but that is an
argument to weigh, not a reason to discount the finding.

### Second MESA round — the severity trade-off was a reference artefact

A second sample of 50 records, seed 20260802, drawn from the 2005 that the
first round did not use (**overlap 0**, asserted by the harness). Three
configurations run on the *same* records so every comparison is paired:
`aasm_v3_rec`, breath at 0.50, breath at 0.55.

The reason for the round was a mistake worth recording. On the strength of
the `Normal → Mild` drift above, strictness was raised to 0.55 — a change
made *after* seeing the round-1 result, and therefore no longer held-out.
This round was meant to confirm it on unseen records. It refuted it.

Against **`aasm15`** (reconstructed from `nsrr_ahi_hp3r_aasm15`: all apneas
plus hypopneas with ≥30% reduction and (≥3% desaturation **or arousal**) —
AASM v3 Rule 1A, r = 0.999 against the published column):

| config | F1 med | F1 mean | prec | recall | bias | MAE | r | severity |
|---|---|---|---|---|---|---|---|---|
| `aasm_v3_rec` | 0.417 | 0.390 | 0.547 | 0.356 | −16.52 | 16.79 | 0.686 | 15/50 |
| **breath @ 0.50** | **0.471** | **0.450** | 0.567 | **0.407** | **−15.78** | **15.96** | 0.838 | **24/50** |
| breath @ 0.55 | 0.441 | 0.431 | **0.574** | 0.378 | −16.93 | 17.03 | **0.841** | 20/50 |

Paired against the previous default: **0.50 improves 36/50 records
(p = 0.0016)**; 0.55 improves 31/50 (p = 0.026).

**0.50 beats 0.55 on F1, recall, bias, MAE and severity — on all three
references.** Severity concordance, the very thing 0.55 was supposed to
protect: 24/50 vs 20/50 (`aasm15`), 28/50 vs 25/50 (`oahi4`), 24/50 vs 21/50
(`oahi3`). Strictness has therefore been returned to **0.50**.

Why the round-1 reading was wrong is visible in the direction of the errors:

```
aasm_v3_rec   Severe→Moderate:10, Mild→Normal:9, Severe→Mild:6, Moderate→Normal:4
breath @0.50  Moderate→Mild:7, Severe→Moderate:7, Severe→Mild:6, Mild→Normal:3
```

Against the rule these profiles actually implement, the problem was never
over-calling — it is substantial **under**-detection (bias −16.5;
`aasm_v3_rec` gets only 15/50 severities right). The breath detector reduces
it: `Mild → Normal` falls from 9 to 3, and severity concordance rises from
15/50 to 24/50. The `Normal → Mild` drift that motivated 0.55 was an artefact
of scoring a Rule 1A profile against a 4% reference.

The general lesson, which cost a wrong parameter change: **when a validation
reference encodes a different rule than the thing being validated, its
error-direction statistics are not interpretable — only the paired
difference is.** The paired result held throughout; the severity reading did
not.

**What is still not established.** Both rounds are MESA. Cross-cohort
transfer is untested — SHHS was considered and set aside on data-quality
grounds — so nothing here shows the result holds on different equipment,
a different scoring era, or a different population. And the absolute level
remains poor: a bias of −16.5 AHI against Rule 1A means both profiles miss a
large share of qualifying events. This work improves the gap; it does not
close it.

## Known issue — blocks enabling the arousal-limb work by default

**The arousal plumbing repair changes `mesa_shhs`, which is a hard
requirement violation, and the cause is the ML re-classifier.**

`mesa_shhs` is the only profile that runs the LightGBM candidate
re-classifier, and `apply_ml_reclassification()` takes the arousal list.
Its highest-gain features are arousal-based (`n_arousals_per_h`,
`n_arousals_within_30s`, `has_arousal_within_5s`). Because of issue #16
those features were **always zero at inference**. Feeding them real values
makes the model reclassify large numbers of rejected candidates:

| mesa-sleep-2408 | main | this branch |
|---|---|---|
| AHI | 30.4 | **54.9** |
| RDI | 30.4 | 84.4 |
| events | 291 | 465 |
| RERAs | 0 | 204 |
| arousals visible | 0 | 372 |

This happens with the arousal limb **disabled** — it is the classifier, not
the limb. It also means the model has a train/serve mismatch that predates
this work: whatever it learned about arousal features, it has never seen a
non-zero one in production. Re-validating or retraining the classifier is a
prerequisite for shipping the plumbing fix, and that is out of scope here.

## Fixed

- **The AASM Rule 1A arousal limb was dead, and so was FRI-based RERA
  detection.** `run_arousal_respiratory_analysis()` returns its events nested
  at `["arousals"]["events"]`, while step 8 read the flat
  `output["arousal"]["events"]` — a key the auto-detection path never
  populated (issue #16). On PSG-IPA SN3: 232 arousals detected, 0 seen.
  `_normalise_arousal_block()` now guarantees the flat key from either
  producer.

  This also repairs two consumers outside scoring: YASAFlaskified's EDF+
  export and event API read the same flat key, so exported EDF+ files
  contained no arousals and the score editor showed none.

  Repairing the plumbing alone already changes **RDI**, because
  `_compute_rera_rdi()` uses the same list — FRI-based RERAs could never
  couple. The golden case shows it: `rdi 0.0 -> 37.9`.

- **`rule1a_arousal_stats`** in the respiratory output: arousals available,
  candidates tested, coupled, qualified, and an explicit `skipped_reason`.
  A limb sitting silently at zero is what let this survive; zero now always
  comes with a reason.

## Added

- **`breath_scoring.py` — a hypopnea detector with the breath as the atom,
  behind the new profile `aasm_v3_breath`** (family `exploratory`, used by
  nothing automatically).

  The existing detector is a signal-processing model: envelope → sliding
  threshold → candidate → validate → classify, with a dozen corrections on
  top. Nearly every correction repairs the same gap with what a scorer
  actually does. This replaces that mismatch rather than patching it
  further, along five shifts:

  1. **The breath is the atom, not the sample.** The AASM speaks of *peak
     signal excursions*. Event boundaries land on breath transitions, and a
     recovery breath breaks the run by itself — so smoothing is gone, and
     with it the merge/split problem that depresses event-F1.
  2. **Two-pass patient calibration.** Pass 1 detects only the
     incontestable events and derives a template — typical depth, typical
     duration, and the patient's **own SpO₂ lag** via cross-correlation
     instead of one fixed 30–45 s window for everybody. Pass 2 judges the
     marginal candidates against that template.
  3. **Graded AASM predicates.** The rule structure stays literally the
     Rule 1A conjunction; what changes is that each threshold gets a
     tolerance instead of being infinitely sharp. A 29% reduction with a 6%
     desaturation should score; 31% with a doubtful 3.0% should not.
  4. **Each event carries `p_scored`** — how well it satisfies the
     conjunction — plus a `criteria` dict with each predicate's
     contribution. The audit trail gets richer, not more opaque.
     *(v0.13.0 shipped this documented as "what fraction of scorers would
     mark this". That was an untested claim and it is wrong; see the v0.13.1
     entry. `p_scored` ranks events, it is not a probability.)*
  5. **Strictness is one axis** instead of three parameter combinations.

  Scope is hypopneas only. Apneas keep the existing detector (F1 0.83–0.93
  on PSG-IPA, with effort-based subtyping); there is no deficit there for
  this to fix.

  **Measured on PSG-IPA (5 recordings, 12 scorers each, legacy matcher).**

  The operating point was chosen with a **rule declared before the sweep**:
  take the strictness where the AHI bias crosses zero, *not* the strictness
  that maximises F1 — the latter would be fitting on the outcome measure.
  That rule lands on 0.50. F1 and percentile are therefore reported as
  non-fitted outcomes.

  | config | F1 median | F1 mean | pct | AHI bias | MAE | severity | hyp |
  |---|---|---|---|---|---|---|---|
  | `aasm_v3_rec` (previous default) | 0.343 | 0.460 | p6 | +1.77 | 1.84 | 4/5 | 208 |
  | breath, strictness 0.35 | 0.447 | 0.522 | p17 | +2.73 | 2.73 | 3/5 | 242 |
  | breath, strictness 0.45 | 0.430 | 0.515 | p17 | +1.13 | 1.13 | 4/5 | 192 |
  | **breath, strictness 0.50** | **0.434** | **0.511** | **p17** | **+0.17** | **0.29** | 4/5 | 163 |
  | breath, strictness 0.55 | 0.426 | 0.510 | p14 | −0.89 | 0.89 | **5/5** | 131 |
  | breath, strictness 0.62 | 0.438 | — | p9 | −1.57 | 1.57 | 5/5 | 110 |
  | breath, strictness 0.70 | 0.396 | 0.495 | p14 | −2.61 | 2.61 | 4/5 | 79 |

  Per recording at strictness 0.50, with the human-human distribution for
  scale (66 pairwise F1s per recording):

  | rec | human median | F1 previous | F1 breath | Δ | pct |
  |---|---|---|---|---|---|
  | SN1 | 0.826 | 0.470 | 0.662 | **+0.192** | p0 → p0 |
  | SN2 | 0.549 | 0.317 | 0.380 | +0.063 | p9 → **p23** |
  | SN3 | 0.948 | 0.886 | 0.897 | +0.011 | p2 → p5 |
  | SN4 | 0.553 | 0.286 | 0.184 | **−0.102** | p17 → p17 |
  | SN5 | 0.556 | 0.343 | 0.434 | +0.091 | p6 → **p18** |

  Better on every aggregate: median F1, mean F1, percentile, bias, MAE;
  severity concordance equal at 4/5. MAE drops from 1.84 to **0.29** AHI
  points.

  The detector diagnostics confirm the mechanism rather than just the
  outcome. The patient-specific SpO₂ lag comes out at 21/12/15/19/17 s —
  plausible and genuinely per-patient, which is the whole point of shift 2 —
  and the template duration is 15–21 s, i.e. actual hypopnea durations.

  **Known limitations.**

  - **SN4 regresses (0.286 → 0.184) and no strictness value fixes it.**
    Of its 24 events with ≥6/12 scorer consensus, **12 never become a
    candidate at all**: at those locations there is no breath-level flow
    reduction of ≥15% sustained over 10 s. Compare SN1, where 33 of 34
    consensus events do produce a candidate. This is a sensitivity ceiling
    in candidate formation, not a threshold that can be turned down. Its
    percentile is unchanged at p17 either way, because SN4's own scorers
    disagree strongly there (pairwise F1 p10 = 0.087, p25 = 0.462) — but
    that is context, not an excuse.
  - **n = 5, and the operating-point rule was applied to the same five
    recordings the result is reported on.** See the MESA held-out result
    below, which was run afterwards with nothing re-tuned.
  - On the recordings where scorers *do* agree, the algorithm remains at or
    below the bottom of their distribution (SN1 p0 against a human median
    of 0.826; SN3 p5 against 0.948).

- **`compute_pre_event_baseline()`** in `signal.py` — the AASM-conforming
  baseline: only breaths in the window *before* the event, breath-amplitude
  based. Stable breathing (CV below threshold) gives the mean amplitude;
  otherwise the mean of the N largest, which is the operationalisation the
  AASM gives when stable breathing cannot be determined. Returns `None`
  rather than a fabricated number when the window holds no usable breathing
  (start of recording, after a gap, or entirely in wake) so the caller can
  fall back deliberately.

- **`rule1a_arousal_enabled` in `PostProcessingRules`** (default **False**)
  and `rule1a_gap_max_breaths` (default 1, the previous hard-coded value).

  The plumbing is repaired, but *enabling* the limb is a behaviour change,
  so it stays off. Measured on PSG-IPA with the limb on:

  | rec | AHI off | AHI on | hypopneas | tested | coupled | qualified |
  |---|---|---|---|---|---|---|
  | SN1 | 8.1 | 9.3 | 32 -> 39 | 74 | 11 | 7 |
  | SN3 | 53.8 | 56.0 | 45 -> 58 | 70 | 20 | 13 |
  | SN5 | 11.4 | 14.8 | 63 -> 87 | 170 | 38 | 24 |

  Coupling is far from universal — 11/74, 20/70, 38/170 — which matches the
  phase-1 finding that only ~17% of rejected candidates have an arousal
  inside the 15 s window. Window and gap are now profile-tunable so phase 4
  can determine them empirically instead of by assumption.

- **`baseline_mode` in `PostProcessingRules`** — `"rolling"` (current) or
  `"pre_event"`. **Every profile keeps `"rolling"`**, including `mesa_shhs`;
  nothing switches until phase 4 justifies it. The rolling baseline stays in
  place for signal quality and the existing visualisations either way.

  Wired into the hypopnea threshold assessment: with `"pre_event"` both
  sides of the ratio come from the breath segmentation (candidate mean
  amplitude vs pre-event baseline), so they measure the same quantity. The
  envelope used by the rolling path is a different scale and is deliberately
  not mixed in. When the baseline is unavailable the event falls back to the
  rolling validation rather than being silently dropped.

  `PSGSCORING_BASELINE_MODE` overrides the profile, so phase 4 can measure
  both settings without mutating profiles (same pattern as
  `PSGSCORING_AROUSAL_DERIVATION`).

  First measurement on PSG-IPA SN1 (reference scorer median AHI 5.96):

  | mode | AHI | events | hypopneas | rejected | local_reduction rejects |
  |---|---|---|---|---|---|
  | rolling | 8.1 | 47 | 32 | 74 | **56** |
  | pre_event | 11.6 | 67 | 52 | 54 | **1** |

  The predicted by-catch is confirmed and it is large: Fix 6
  (`local_reduction`) drops from 56 rejections to 1, i.e. it becomes almost
  entirely redundant once the baseline is measured before the event instead
  of around it. Fix 1 and Fix 6 are not removed here — phase 4 measures what
  they still contribute first.

  AHI rises on this recording, away from the scorer median. That is not by
  itself a verdict: the decision measure for this work is event-level F1
  against human scorers and the percentile within the inter-scorer
  distribution, not AHI bias, precisely because a better bias can come from
  errors that cancel. Phase 4 settles it.

  Why it matters: `compute_dynamic_baseline()` uses a *centred* 5-minute
  window, so it includes the recovery hyperpnoea that follows an event. That
  inflates the baseline and shrinks the measured reduction. Fix 1 and Fix 6
  are patches on precisely that design.



**Rename with aliases — no behaviour change** (golden regression green;
PSG-IPA output byte-identical).

## Changed

- **The arousal qualifier is AASM Rule 1A, not 1B.** The AASM places the
  arousal criterion in Rule 1A (">=30% flow reduction AND (>=3% desaturation
  OR arousal)"); Rule 1B is the >=4%-desaturation variant that explicitly
  *excludes* arousals. The code had it the other way round.

  | was | is |
  |---|---|
  | `reinstate_rule1b_hypopneas()` | `reinstate_rule1a_arousal_hypopneas()` |
  | event `classify_detail.rule = "1B_arousal"` | `"1A_arousal"` (+ `rule_legacy`) |
  | event flag `rule1b` | `rule1a_arousal` |
  | `respiratory.rule1b_reinstated` | `respiratory.rule1a_arousal_reinstated` |

## Deprecated (kept working)

Every old name is retained as an alias and is still emitted in the output.
This is not cosmetic caution — three consumers depend on them:
YASAFlaskified imports `reinstate_rule1b_hypopneas` in `pneumo_analysis.py`
and reads `rule1b_reinstated` in both `generate_pdf_report.py` and
`generate_psg_report.py`, and `psgscoring.ml_classifier` uses the candidate
flag `rule1b` as the trained LightGBM feature `is_rule1b` — dropping it
would silently deprive the model of an input. `tests/test_rule1a_arousal_naming.py`
pins all of it.

Note this rename does not by itself make the arousal limb fire; that is a
separate plumbing defect (issue #16).
# v0.12.3 — 2026-07-29 — NSRR nasal pressure detected as flow_pressure, not pulse

**Channel-detection fix — no scoring-logic change** (golden regression byte-identical;
PSG-IPA output byte-identical, md5 5ddbcf42360d965cd20ce1b8796221da before and after).

## Fixed

- **`Pres` (the NSRR/MESA/SHHS nasal-pressure channel) was claimed by the `pulse` role.**
  Channel matching is case-insensitive substring, first-match-wins per role, so pattern
  *order* is semantics. The `pulse` list contained `"pr"` — a substring of `"pres"` — and
  no `flow_pressure` pattern matched a bare `"Pres"`. Two silent consequences on MESA:
  `flow_pressure` stayed empty, so `_resolve_flow_channels()` ran **both** apnea and
  hypopnea detection on the thermistor (amplitude ~1500× smaller than the nasal pressure);
  and the `pulse` role took the nasal pressure instead of `"HR"`.
  `"pres"` is now in `flow_pressure` (deliberately *after* the specific patterns, so an
  explicit `"Nasal Pressure"` still wins) and `"pr"` moved to the end of `pulse`.
  MESA now resolves `flow_pressure=Pres`, `flow_thermistor=Therm`, `pulse=HR`.

Recordings that already resolved their channels correctly are unaffected — PSG-IPA has
neither a `Pres` nor an `HR` channel, and its full 5-recording output is byte-identical.
On MESA the hypopnea F1 barely moves (median 0.304 → 0.321) even though the hypopnea
sensor switches off a 1500× smaller channel: detection normalises, so amplitude scarcely
matters. The value is correctness of sensor assignment, not a metric gain.

# v0.12.2 — 2026-07-27 — docs: absolute DISCLAIMER link (fix dead link on PyPI)

**Docs-only — no code change** (golden regression byte-identical).

- README: the `DISCLAIMER.md` link is now absolute
  (`https://github.com/bartromb/psgscoring/blob/main/DISCLAIMER.md`). Relative links
  break on the PyPI project page (404); this release re-renders the README on PyPI with
  the working link. No API or behaviour change.

# v0.12.1 — 2026-07-26 — ventilatory burden made breath-based (fixes over-count)

**Output-additive — no AHI/OSAS-grade change** (golden regression byte-identical).

## Fixed

- **Ventilatory burden is now breath-based.** v0.12.0 computed VB as the *time-fraction*
  of the normalized flow envelope below 0.5, which over-counted heavily (the smoothed
  Hilbert envelope dips between breaths even during normal breathing, and the baseline is
  the 95th percentile) — a real severe-OSA recording gave an implausible **82.9 %**.
  VB is now the **proportion of breaths whose peak amplitude is < 50 % of the eupneic
  baseline** (per breath, using `output["respiratory"]["_breaths"]`), matching the AJRCCM
  2023 definition and excluding inter-breath troughs. Signature:
  `compute_ventilatory_burden(flow_norm, sf_flow, breaths, hypno=None, threshold=0.5)`.

# v0.12.0 — 2026-07-26 — VB recalibration + saturation bands + arousal-aetiology fix

**Output-additive — no AHI/OSAS-grade change** (golden regression byte-identical).
Three report-driven fixes found while verifying a real v0.11.0/v0.15.0 report.

## Changed

- **Ventilatory burden recalibrated to the validated metric.** `summary["ventilatory_burden"]`
  is now **the percentage of sleep with airflow < 50% of the eupneic baseline** (proportion
  of "small breaths"), per *Ventilatory Burden*, AJRCCM 2023;208(11):1153 & 1216 — a bounded
  0–100% measure that predicts CV/all-cause mortality (normative ≈ ≤25%). Replaces the earlier
  unbounded %·min/h area integral (which produced implausible values, e.g. ~1074). Computed on
  the normalized flow envelope restricted to sleep. `compute_ventilatory_burden` signature is
  now `(flow_norm, sf_flow, hypno=None, threshold=0.5)`; `VB_NORMATIVE_MAX = 25.0` exported.
- **Arousal-aetiology indices now reconcile with the arousal index.** `respiratory_arousal_index`
  + `spontaneous_arousal_index` are derived by splitting `arousal_index` by aetiology fraction,
  so they **sum to the arousal index exactly** (previously divided the raw counts by a
  differently-computed TST, so resp + spont < AI). `plm_arousal_index` is reported as a subset
  of the spontaneous group.

## Added

- **Time-in-saturation-bands** in the SpO2 summary: `time_95_100_min`/`pct_95_100`,
  `time_90_95_min`/`pct_90_95`, `time_80_90_min`/`pct_80_90`, `time_70_80_min`/`pct_70_80`,
  `time_below_70_min` (the report's band table read these keys, which did not exist → all 0.0).

All output-additive; AHI/OSAS/CSAS grade and arousal index unchanged.

# v0.11.0 — 2026-07-26 — AASM v3 clinical enrichments (dual-AHI, CSR density, arousal aetiology)

**Output-additive — no AHI/OSAS-grade change** (golden regression byte-identical).
Derived from the manual re-read in `docs/aasm/` (AASM Scoring Manual v3, 2023).

## Added

- **Dual AHI (A1)** — `summary["ahi_dual"]`: AASM v3 hypopnea **Rule 1A** (≥30% flow +
  ≥3% desat OR arousal — the recommended standard) and **Rule 1B / CMS** (≥4% desat)
  AHI side by side, with severity per rule. Reuses the existing 3-profile confidence
  pass (`standard`=aasm_v3_rec, `strict`=aasm_v3_strict); no extra detection pass.
- **Cheyne-Stokes density criterion (A2)** — `cheyne_stokes` now also reports the AASM
  G.1(b) criterion: `central_events_per_h`, `monitoring_hours`, `density_criterion_met`
  (≥5 central apneas/hypopneas per hour over ≥2 h), and `criteria_met` (periodicity
  G.1(a) **and** density G.1(b)).
- **Arousal aetiology indices (A3)** — `arousal["summary"]`: `respiratory_arousal_index`,
  `spontaneous_arousal_index`, and `plm_arousal_index` (+ `n_plm_arousals`) per hour of
  sleep (AASM V.A Note 4). Counts existed; now exposed as clinical indices.
- **Hypopnea criterion string (A5)** — `meta["hypopnea_criterion"]`: the exact scoring
  rule in words (AASM v3 VIII.D Note 1 requires the criterion be stated in the report).
- **Hypoventilation scope statement (A6)** — `summary["hypoventilation"]` explicitly
  marks it *not assessed* (no PCO2/capnography channel), rather than silently omitting it.

## Changed

- **Apnea/hypopnea max-duration cap (A4)** — the split cap stays profile-default (90 s /
  60 s → **byte-identical**) but is now overridable per site via
  `PSGSCORING_APNEA_MAX_DUR_S` / `PSGSCORING_HYPOPNEA_MAX_DUR_S` (AASM imposes no maximum
  apnea duration; splitting a genuinely long *central* apnea over-counts). Apneas sitting
  at the cap are flagged in `summary["n_apneas_at_max_dur"]`.

All of the above are output-additive; the AHI, OSAS/CSAS grade and arousal index are
unchanged from v0.10.0.

# v0.10.0 — 2026-07-25 — clinical phenotypes (POSA, REM-predominant) + ventilatory burden

**Output-additive — no AHI/OSAS-grade change** (golden regression byte-identical;
validated on PSG-IPA: SN1 AHI 8.1 unchanged).

## Added

- **Phenotype flags** in the respiratory summary (`summary["phenotypes"]`), derived
  from the already-computed position + REM/NREM indices (`_compute_phenotypes()`):
  - **Positional OSA (Cartwright):** supine AHI ≥ 2× non-supine AHI, with OSA present
    (AHI ≥ 5) and ≥ 30 min sleep in both the supine and non-supine groups. Reports
    supine/non-supine AHI + ratio + positional-therapy candidacy (non-supine AHI < 5).
    Requires a body-position channel.
  - **REM-predominant OSA:** REM-AHI ≥ 2× NREM-AHI, with ≥ 30 min REM.
- **Ventilatory burden** (`summary["ventilatory_burden"]`, %·min/h;
  `compute_ventilatory_burden` in new module `ventilation.py`): total event-associated
  airflow deficit relative to the eupneic baseline — pairs with the hypoxic burden as a
  cardiovascular-risk duo beyond the AHI. Method after Labarca et al. 2023.
  ⚠️ The exact parameterization/scale should be confirmed against Labarca 2023 before
  clinical interpretation (no reference range asserted yet).

Both are output-additive; the AHI and all existing indices are unchanged.

# v0.9.0 — 2026-07-25 — multi-derivation arousal detection is now the DEFAULT

**Behaviour change (clinical profiles only).** Arousal detection now defaults to
**multi-derivation** (central + occipital + frontal, event-level union + EOG-reject)
for all `family="clinical"` profiles (`aasm_v3_rec`/`_strict`/`_sensitive` and their
aliases). `single` is retained as an explicit option.

## Changed

- **Default arousal mode per profile family** (`constants.py`,
  `AROUSAL_DERIVATION_MODE`): **clinical → `multi`**, **dataset → `single`**
  (`mesa_shhs` and any NSRR-reproduction profile stay single-channel — their
  reference was scored on one central derivation, so this keeps MESA/SHHS
  reproduction byte-identical), legacy/exploratory → `multi`.
- Choose explicitly at any time with env `PSGSCORING_AROUSAL_DERIVATION=single|multi`
  (overrides the profile) or the profile field. With < 2 usable EEG derivations,
  `multi` degrades to `single` automatically.

## Impact & reproducibility

- **The AHI is unchanged** by this flip on the validation cohort (arousals
  contribute 0 arousal-only events; the golden cases carry no EEG so they are
  unaffected). The **arousal index rises** (multi is more sensitive: PSG-IPA
  sens 0.38 → 0.47 vs the scorer consensus).
- **For exact single-channel reproduction of the paper's PSG-IPA numbers**, set
  `PSGSCORING_AROUSAL_DERIVATION=single`. MESA reproduction is unaffected
  (`mesa_shhs` is a dataset profile → already single).
- Rationale: clinicians score arousals by scanning the whole montage; multi
  (union + EOG-reject) is the more sensitive, more human-like operating point.
  Note the Fase-3 finding stands — the gain is *generic sensitivity*, not an
  occipital-specific recovery (see v0.8.1 notes) — so this is a deliberate
  sensitivity-first product choice, not an accuracy improvement.

# v0.8.1 — 2026-07-25 — multi-derivation arousal detection (opt-in; default unchanged)

**No default behaviour change — `single` mode is byte-identical to v0.8.0.**

## Added

- **Multi-derivation arousal scoring** (`arousal.detect_arousals_multi` + event-level
  `_union_arousals`; `pipeline._pick_eeg_multi`). Runs the mature single-channel
  detector independently on central (C4-M1) + occipital (O2-M1) + frontal (F4-M1)
  and unions overlapping events, keeping per-channel validation (spindle exclusion,
  K-complex, REM-EMG) intact. Events gain `derivation`/`derivations` provenance.
- Opt-in via env `PSGSCORING_AROUSAL_DERIVATION=multi` (or a future profile field
  `AROUSAL_DERIVATION_MODE`). **Default stays `single`**; with < 2 usable
  derivations it degrades to single automatically.
- **Hard invariant** (unit-tested): multi with one derivation == `detect_arousals`
  byte-for-byte.

### Fase 2 — human-like refinement (also opt-in)

- **Per-channel detection thresholds are now parameters** (`detect_arousals(...,
  ratio_thresh=, abrupt_thresh=)`), replacing module-global mutation — concurrency-safe
  (8 workers + the multi-derivation loop) and enabling per-derivation calibration.
  Defaults unchanged → **byte-identical** (unit-tested), incl. the LGBM hybrid path.
- **`detect_arousals_multi(..., per_channel_thresh=, eog_data=, eog_reject=)`**: each
  derivation can run at its own calibrated threshold, and an **EOG-based reject drops
  occipital-only events that coincide with a large eye movement** (EOG-doorslag) —
  mirroring a scorer cross-referencing the EOG. Cross-channel-confirmed events are
  never touched. All default-off.

## Validation (PSG-IPA, 5 recordings × 12 scorers, consensus ≥6/12)

- **Per-channel sweep vs scorer consensus** — best single-channel F1: C4-M1 0.28,
  O2-M1 0.32, **F4-M1 0.41** (frontal > occipital > central), i.e. the standard
  single central derivation is the *weakest* for arousals. F1 peaks near the current
  defaults; frontal benefits from ratio 2.5.
- **Payoff (single vs multi-default vs multi-tuned+EOG-reject):**
  sens **0.38 → 0.53 → 0.47**, PPV **0.36 → 0.30 → 0.32**, F1 **0.37 / 0.38 / 0.38**.
  The tuning+EOG-reject **cuts ~16% of the naïve-union detections (mostly false
  positives) and recovers PPV while keeping most of the sensitivity gain** — but no
  configuration dominates on F1 (arousal-vs-consensus agreement is fundamentally
  modest, consistent with low inter-scorer reliability).
- **Net:** multi (esp. tuned+EOG-reject) is a defensible, more sensitive, more
  human-like operating point (its value is *sensitivity* — catching arousals the
  single central channel misses — not overall accuracy). The **AHI is unaffected**
  (0 arousal-only events on this cohort). Kept **opt-in**; no default flip. The
  scorer consensus is itself an imperfect reference (low inter-scorer agreement),
  so F1 understates true performance.

### Fase 3 — pipeline-wired + the occipital question settled

- **Multi mode is now fully reachable through `run_pneumo_analysis`** (not just the
  low-level API): set `PSGSCORING_AROUSAL_DERIVATION=multi` (or profile field
  `AROUSAL_DERIVATION_MODE`). In multi mode the pipeline unions the available
  central/occipital/frontal derivations **and applies the EOG-reject by default**
  (picks an EOG channel; disable with `PSGSCORING_AROUSAL_EOG_REJECT=0`). New
  `pipeline._pick_eog`. **Default stays `single` → byte-identical** (re-verified:
  SN1 8.1/32 unchanged).
- **The original occipital hypothesis is NOT supported by the data.** Per-derivation
  recall on scorer-consensus arousals split by where scorers marked them
  (posterior/occipital-involving vs anterior): C4-M1 **0.29 post / 0.23 ant** (no
  posterior blind spot — central is if anything slightly *better* posteriorly),
  O2-M1 **0.31 / 0.31** (occipital is not specifically better at occipital-evident
  arousals), **F4-M1 0.39 / 0.46 (frontal is the strongest single channel)**, multi
  0.50 / 0.55 (a *generic* sensitivity gain across both classes). So multi-derivation
  is a coverage/sensitivity choice, not a fix for a central occipital blind spot.
- **Design decision:** ship multi as a well-engineered, EOG-aware, **opt-in
  high-sensitivity mode**; keep single-central as the reproducible default. No
  default flip is justified by the current evidence (F1 does not improve; the AHI is
  unchanged; occipital adds no specific class of arousals).

# v0.8.0 — 2026-07-25 — arousal & RERA detection moved in-package (byte-identical)

**Structural — no scoring change. AHI / ODI / events / hypoxic burden byte-identical.**
Validated on the real PSG-IPA cohort (5 recordings): AHI, arousal counts and
arousal index unchanged to the digit (SN1 8.1/32 arousals, SN3 53.8/185, …;
0/516 AH-events arousal-only — the AHI is desaturation-driven).

## Added / Changed

- **New module `psgscoring.arousal`** — EEG arousal detection, RERA detection,
  flow-limitation detection and respiratory-arousal coupling now live **inside
  psgscoring**. Ported verbatim from YASAFlaskified `arousal_analysis.py` so the
  library is self-contained; `run_pneumo_analysis` no longer depends on an
  external, optionally-importable module. New public API: `detect_arousals`,
  `detect_reras`, `run_arousal_respiratory_analysis`.
- `pipeline.py` imports the detector in-package (`from .arousal import …`) instead
  of the fragile `from arousal_analysis import …` optional import.
- Added `from __future__ import annotations` to the ported module — fixes a
  Python 3.9 crash (PEP 604 `X | None` unions evaluated at def-time).
- Optional LightGBM arousal re-classifier carried over; model path/env made
  psgscoring-neutral (`PSGSCORING_AROUSAL_LGBM[_MODEL/_THRESHOLD]`, old
  `YASAFLASKIFIED_*` env still honoured). Default off → pure rule-based, identical.
- YASAFlaskified `arousal_analysis.py` becomes a thin re-export shim
  (requires psgscoring ≥ 0.8.0; deploy the pair together).

This lands the groundwork for **multi-derivation arousal scoring** (central +
occipital + frontal), which will be added as an opt-in mode on top.

# v0.7.6 — 2026-07-22 — fix: hypoxic-burden TST denominator excludes invalid SpO2

**Numerics-changing — hypoxic burden only; no AHI/ODI/event change.** Closes #2
(PR #3).

## Fixed

- **Hypoxic-burden TST denominator now excludes invalid SpO2 samples**
  (`spo2.py`, `compute_hypoxic_burden`). The per-event desaturation-area
  numerator already drops NaN / out-of-[50,100] samples, but the TST
  denominator was built from the hypnogram alone (`sleep_mask`), so sensor
  dropouts during sleep inflated the denominator and **deflated** the burden.
  `tst_h` now uses `sleep_mask & ~np.isnan(spo2)`, matching the de Chazal
  `calcHB.m` reference (`HourSleep`). On a recording with ~32 min of in-sleep
  SpO2 < 50 the reference TST shifts 7.20 h → 6.67 h (higher, correct HB).
- Only `hypoxic_burden` changes, and only on recordings with invalid in-sleep
  SpO2; recordings with clean oximetry are byte-identical (`~np.isnan` is all
  True). AHI / ODI / events / all other fields are untouched. Golden 6/6
  unchanged (synthetic cases have no in-sleep dropouts). New unit test
  (`test_spo2_numeric.py`) asserts HB rises by exactly the TST reduction.
- **MESA A/B (2026-07-22):** on the same 16-recording sample, `hypoxic_burden`
  changed on 5 recordings — all **raised** (e.g. 156.18→167.52, 17.06→18.34),
  never lowered — while AHI/events stayed byte-identical, confirming the
  denominator correction behaves as intended on real data.

# v0.7.5 — 2026-07-22 — fix: RERA/RDI dropped on Cheyne-Stokes nights

> Stacks on **v0.7.4** (the output-preserving robustness/test PR); merge that first.

**Numerics-changing — but strictly additive: no AHI, event, or SpO2 number
changes.** On recordings where Cheyne-Stokes respiration is detected, the RERA
index, RDI, and REM/NREM AHI silently vanished from `summary`; they are now
retained.

## Fixed

- **RERA/RDI/REM-NREM AHI no longer wiped on CSR-positive nights**
  (`pipeline.py`). `_compute_rera_rdi` (step 8b) wrote `rera_index` / `rdi` /
  `n_rera` / `rem_ahi` / `nrem_ahi` into the respiratory summary, but the
  Cheyne-Stokes "Fix 3" step then replaced `output["respiratory"]["summary"]`
  wholesale with a fresh `_compute_summary()` — dropping every one of those keys
  whenever `csr_detected` was true. `_compute_rera_rdi` now runs **after** the
  CSR summary recompute, so the keys survive. Non-CSR nights are unaffected
  (the CSR block does not fire, and nothing between the old and new call
  position touches the summary → byte-identical); CSR nights now report RDI et
  al. instead of `None`.

## Validation

- The fix is a pure statement reorder. `_compute_rera_rdi` only ever **reads**
  `ahi_total` and **writes** the RERA-family keys, so it cannot move any AHI /
  OAHI / event / SpO2 value — RDI is derived as `ahi_total + rera_index`, never
  the reverse.
- **Golden harness (6/6):** the 3 CSR-positive synthetic cases show exactly one
  changed field — `resp.rdi: null → value` (= `ahi_total`, no arousals) — with
  `ahi_total`, every event, and all other summary fields byte-identical;
  re-blessed accordingly. The 3 non-CSR cases are unchanged.
- **PSG-IPA reproducibility:** byte-identical (clinical cohort, non-CSR).
- **New regression tests** (`test_rera_csr_ordering.py`, 4 cases) build a
  CSR-positive synthetic recording (periodic apneas) and assert the full RERA
  family — including non-trivial values with arousals (`rera_index > 0`) and REM
  (`rem_ahi` populated) — survives to the final output.
- **MESA empirical A/B — CONFIRMED (2026-07-22).** Ran `scripts/ab_rera_csr.py`
  on a 16-recording MESA sample (9 CSR-positive) with the validated `score_mesa`
  harness (validation-mode tuning, NSRR hypnogram + arousals): 0.7.3 vs 0.7.6.
  Result — **0 invariant fields moved** (`ahi_total` / `ahi_incl_uncertain` /
  `n_events` byte-identical on all 16), and the RERA family restored (None→value)
  on exactly the 9 CSR-positive recordings (45 keys). Confirms the fix is
  strictly additive on real clinical data.

# v0.7.4 — 2026-07-22 — robustness & test coverage (output-preserving)

Code-review follow-up. **No scoring changes** — golden harness + PSG-IPA
reproducibility byte-identical; full suite green (129 passed).

- **Graceful degradation for ancillary steps.** A failure in any single
  ancillary analysis (SpO2, position, heart rate, snore, PLM, arousal) now
  degrades to a `{"success": False, "error": …}` result via a new `_run_step`
  helper instead of aborting the whole `run_pneumo_analysis`, matching the
  pattern the interval/ML/CSR/hypoxic-burden/post-processing steps already used.
- **`apply_ml_reclassification` honours its docstring contract.** The whole
  featurise → predict → sort body is guarded, so a malformed candidate (missing
  `onset_s`), a booster shape/NaN mismatch, or the onset sort now falls back to
  the rule-based result rather than crashing.
- **`ecg_effort.detect_r_peaks` guards low sample rates.** The 5–30 Hz QRS
  bandpass would raise for `sf ≤ 60` (Nyquist ≤ 30); it now returns no peaks,
  which also stops `compute_tecg` crashing on low-rate ECG. Never fires on real
  PSG ECG (128–512 Hz), so output is unchanged.
- **Removed dead recomputation in `detect_respiratory_events`.** The initial
  hypopnea mask (peak + envelope + OR, plus a full breath-amplitude pass) was
  computed and then discarded — only the post-apnea-corrected mask is consumed.
  Deleting it removes redundant work per profile with byte-identical output.
- **New numeric unit tests** (`test_spo2_numeric`, `test_respiratory_core`,
  `test_postprocess_numeric`, `test_crash_safety`) so ODI / T90 / hypoxic
  burden / breath segmentation / dynamic baseline / CII and the crash-safety
  guards are checked on every `pytest tests/` — previously the only numeric
  coverage was the env-gated golden test.
- **Packaging:** ship `py.typed` (PEP 561) so downstream mypy/pyright see the
  inline annotations.
- Housekeeping: `signal_quality_channels.py` gets its own logger name (was
  colliding with `signal_quality.py`), a correct filename docstring, and the
  fictitious "v0.8.30" stamp removed; dropped a redundant inline
  `preprocess_flow` re-import in `pipeline.py`.

# v0.7.3 — 2026-06-09 — documentation / terminology

Docs and terminology only — **no code or scoring changes** (golden harness
byte-identical; full suite green).

- **Dropped the "AASM 2.6" version label throughout** (project description, README,
  DISCLAIMER, docstrings and code comments). The library supports AASM v1 / v2 / v3
  via scoring profiles (default `aasm_v3_rec` = AASM 2023), so pinning the prose to the
  2.6 manual was inaccurate. Text now reads "AASM-compliant" / "AASM" / the specific
  profile name.
- **Refreshed the developer handbook** (`docs/developer_handbook.md`) to the current
  state: v0.7.2 / YASAFlaskified v0.12.4, psgscoring now pip-installed (not bundled),
  the git → PR → CI → GitHub-Release/OIDC → rsync workflow, the v0.7.x profile set +
  3-profile AHI interval + v0.7.2 shared-preprocessing speed-up, and the paper-v37
  validation table. Authors section updated.

# v0.7.2 — 2026-06-07 — performance (shared preprocessing)

**No scoring changes — output is byte-identical.** Speeds up `run_pneumo_analysis`
by ~1.8–2.0× by removing redundant work in the 3-profile AHI confidence interval.

## Performance

- **Preprocessing computed once across the AHI-interval profiles** (`pipeline.py`,
  `respiratory.py`). The confidence interval ran `detect_respiratory_events` 4×
  per recording (primary + strict/standard/sensitive), each redoing the full
  preprocessing: Hilbert envelopes (~43 s), dynamic baseline / rolling percentile
  (~24 s), position-change detection (~10 s), MMSD, effort/SpO₂ baselines, breath
  detection. All of this depends only on the raw signals and on baseline params
  that are identical across the three interval profiles, so it is byte-identical
  across the reruns. A shared `_precomputed` cache now computes it once; only
  per-profile event qualification reruns. Baseline-dependent caches are keyed by
  `(BASELINE_WINDOW_S, BASELINE_PERCENTILE)` so a non-v3 primary stays correct.
- **Position changes computed once** (part of the cached baseline block).
- **Fixed an interval reuse bug:** the primary result was matched against the
  legacy alias (`standard`), so canonical names (`aasm_v3_rec`) never matched and
  the primary was needlessly re-scored a 4th time. Now matched on `_PROFILE_NAME`.

Validated byte-identical (every summary field, AHI interval, and individual
event matches the previous output exactly) against:
- the golden harness (6/6) and the full unit suite (104 passed, 11 skipped);
- the **MESA q7 holdout** — all 92 locally-available recordings, via the
  `score_one_mesa` validation path (arousal injection + artifact epochs +
  the `aasm_v2_rec` profile with the validation-mode tuning);
- the **PSG-IPA clinical cohort** (SN1–SN5, profile `standard` = `aasm_v3_rec`);
- the `cms_medicare` param-keyed cache path (MESA id 301).

Measured speedup on the analysis: MESA id 301 125.8 s → 62.5 s (2.0×),
id 33 90.5 s → 49.7 s (1.8×).

# v0.7.1 — 2026-06-03 — documentation

Docs-only release — **no code or scoring changes** (identical scoring to
v0.7.0). Corrects the README / PyPI project description:

- Quick Start uses the canonical profile name `aasm_v3_rec` (not the
  deprecated `standard` alias) and the correct `results["ahi_interval"]` key.
- Architecture section: drop the removed `pipeline_profiles` module, add
  `ml_classifier`; refresh module/test/line counts (17 submodules, 115 tests).
- PSG-IPA validation figure corrected to the validated mean |ΔAHI| = 1.8/h
  (Pearson r = 0.997).

# v0.7.0 — 2026-06-03 — Tier-1 scoring-accuracy fixes

These fixes correct over-detection on degraded signal and profile-specific
counting. They were **validated to leave both paper cohorts byte-identical**
(PSG-IPA clinical and the MESA q=7 holdout: 0 recordings changed vs v0.6.2)
and to have negligible real-data impact on the MESA q∈[2,4] POOR cohort
(1/94 recordings changed; agreement vs the scorer reference unchanged) — i.e.
they fix genuine edge-case bugs without moving any validation number. The
large effects on synthetic stress cases (a literal flat-line channel; the
`cms_medicare` profile) do not arise on real recordings. Clean-signal output
is unchanged.

## Fixed

- **Dead/flat signal regions no longer scored as apneas**
  (`respiratory.py` `_detect_signal_gaps`). A frozen/dead channel had
  `flow_norm ≈ 0` across the flat span and scored as back-to-back apneas;
  the span itself is now excluded from event detection (not just the
  post-gap recovery ramp). Large effect on POOR-quality / dropout
  recordings — golden `poor_quality`: 14→2 events, AHI 88.4→12.6;
  `flat_dropout`: 5→4, AHI 31.6→25.3.
- **CMS / AASM-v1 profiles no longer reinstate arousal-only hypopneas**
  (`pipeline.py` Rule 1B gated on `DESAT_OR_AROUSAL`). `cms_medicare` and
  `aasm_v1_rec` score hypopneas on desaturation only; arousal-coupled
  reinstatement is now skipped for them — golden `cms_arousal`:
  `rule1b_reinstated` 2→0, AHI 37.9→25.3. No effect on `aasm_v3_*` /
  `mesa_shhs` (`DESAT_OR_AROUSAL=True`).
- **Hypopnea baseline honors the profile window/percentile on the
  mixed-sample-rate / RIPsum-fallback path** (`respiratory.py`
  `_setup_hypop_channel`). Previously reverted to the 300 s / 95th
  defaults instead of e.g. `mesa_shhs`'s 120 s / 85th. Affects `mesa_shhs`
  on mixed-sample-rate / degraded-nasal recordings — needs real-data
  validation (single-sfreq synthetic cases do not exercise it).
- **SpO2 sensor-dropout gaps no longer manufacture desaturations**
  (`spo2.py` `detect_desaturations`). NaN gaps were filled with a constant
  95%, creating a fake plateau at gap edges that registered as a
  desaturation; gaps are now interpolated for smoothing and the dropout
  regions are excluded from detection.
- **Mixed-apnea decomposition is NaN-safe** (`postprocess.py`). A NaN in
  the effort segment made `np.max` → NaN, silently defeating central-portion
  detection; now uses `np.nanmax` with a finite guard. (Type label only;
  no AHI change.)

## Validation status

#1 and #3 are quantified by the golden harness (above). #4 and #6 are
implemented but need real `mesa_shhs` / MESA validation. Before any 0.7.0
release: run the q7 cohort on 0.7.x vs 0.6.2 (golden_snapshot / score_mesa)
and re-run the PSG-IPA clinical validation to confirm the clinical headline
is unchanged.

# v0.6.2 — 2026-06-03

Dual AHI reporting for unsubtyped apneas. No change to event detection or to
any existing field; adds two summary keys.

## Added

- **`summary["ahi_incl_uncertain"]`** — scorer-calibrated AHI that also counts
  apneas the effort-based classifier could not subtype (`type == "uncertain"`,
  typically when the RIP/effort signal is degraded). `ahi_total` continues to
  exclude these (conservative, flag-for-review). On the MESA q=7 holdout the
  inclusive index is essentially unbiased against the NSRR scorer reference
  (bias ≈ −0.3/h), whereas `ahi_total` runs ≈1.5/h lower; both are now
  reported so downstream consumers can choose.
- **`summary["n_uncertain_apnea"]`** — count of detected-but-unsubtyped apneas
  (the events that distinguish the two indices), for transparency / review.

## Notes

- An `uncertain` apnea is still a scored apnea; AHI by definition counts every
  apnea regardless of obstructive/central/mixed subtype. Clinical (non-ML)
  profiles produce no `uncertain` apneas, so `ahi_incl_uncertain == ahi_total`
  there and PSG-IPA reproducibility is unchanged.

# v0.6.1 — 2026-06-03

Robustness and reproducibility patch. Fixes two POOR-quality scoring
crashes, a Python 3.9 import regression, and a signal-quality grading
bug; adds a golden-output regression harness. **Clinical AASM profiles
are unchanged** — PSG-IPA reproducibility (10/10) is intact and the
fixes only recover recordings that previously crashed; they do not
alter the scoring of any recording that already produced a result.

## Fixed

- **Rule 1B reinstatement crash (`KeyError: 'stage'`).** The
  stable-breathing rejection filter stored rejected hypopnea
  candidates without `stage`/`epoch`, so `reinstate_rule1b_hypopneas`
  crashed whenever such a candidate coincided with an arousal — i.e.
  any recording scored with EEG arousal detection (the normal clinical
  path). The Rule 1B call had no `try/except`, so the whole analysis
  aborted.
- **ML re-classification crash (`KeyError: 'type'`).** The `mesa_shhs`
  LightGBM re-classifier promotes pooled candidates into the accepted
  list, but rejected hypopnea candidates carried no `type` key, so a
  promoted one was type-less and crashed `_compute_summary`. Recovered
  ~64% of the MESA q∈[2,4] graceful-degradation cohort (30/100 → 94/100
  scored). No effect on ML decisions (feature extraction already
  defaulted `type` to `"hypopnea"`).
- **Python 3.9 import crash.** `signal_quality_channels.py` used a
  PEP 604 `list | None` annotation without
  `from __future__ import annotations`, making the module unimportable
  on Python 3.9 — silently disabling per-channel quality grading
  (`channel_quality: "unknown"`) for all 3.9 users.
- **`_count_flat_samples` overcount.** The flat-sample count was
  inflated by ~the window length (a `/len*len` cancellation), so
  `flat_pct` could exceed 100% and clean channels were mis-graded
  `"poor"`. Metadata only — no effect on AHI.
- **Robustness guards.** Empty-flow input in `_detect_signal_gaps`;
  out-of-range event epoch in `analyze_position` (one bad epoch no
  longer drops the whole per-position summary).

## Added

- `tests/test_golden_output.py` + `tests/golden/` — golden-output
  regression harness (6 deterministic synthetic cases; gated behind
  the `PSGSCORING_GOLDEN` env var, out of the default CI matrix).
- `scripts/golden_snapshot.py` — bless/check pipeline output on real
  EDFs with a per-recording AHI-delta table.
- `tests/test_rejected_candidate_invariant.py` — guards the invariant
  that rejected candidates carry `type`/`stage`/`epoch`.

## Changed

- `_types.PLMSummary` field names aligned with `analyze_plm` output
  (`plm_index`, `n_resp_associated`, …); added the missing keys.
- Removed the unreferenced `pipeline_profiles.py` stub (stale
  duplicate `run_pneumo_analysis`).

# v0.6.0 — 2026-05-05

LightGBM candidate-level re-classifier as an optional post-detection
step. Trained on the q$\geq$5∖q=7 stratum of the MESA cohort
(n=653 recordings, ~210k labelled candidates) with 5-fold group-CV
by mesaid; held out q=7 entirely from training. Default in the
`mesa_shhs` profile; clinical profiles leave it None and skip the
step (PSG-IPA paper-v31 reproducibility 10/10 pass).

## Added

- `psgscoring/ml_classifier.py` — module providing `load_booster`,
  `apply_ml_reclassification`, and a runtime feature extractor
  (`_extract_candidate_features`) mirroring the training-time
  `build_lightgbm_dataset.extract_features` (32 features per
  candidate: event-intrinsic, sleep-stage, cluster context,
  recording-level, surrounding arousals).
- `PostProcessingRules.ml_classifier_path: str | None = None`
  — path (absolute or relative to package data dir) to a LightGBM
  booster file. None disables the step (default for clinical).
- `PostProcessingRules.ml_threshold: float = 0.65`
  — bias-near-zero operating point established on q=7 holdout.
- `psgscoring/data/lightgbm_v06_q7holdout.txt` — shipped 810 KB
  trained model. Top-5 features by gain: `desaturation_pct`,
  `n_arousals_per_h`, `stage_r`, `time_to_next_event_s`,
  `n_arousals_within_30s`.
- `pyproject.toml` — `ml` extra (`pip install psgscoring[ml]`)
  installs `lightgbm>=3.0`; package-data section ships
  `data/*.txt`.
- `pipeline.py` — Step 8a (between Rule 1B and RERA/RDI) calls
  `apply_ml_reclassification` when profile sets a path.

## Changed

- `mesa_shhs` profile sets
  `ml_classifier_path="data/lightgbm_v06_q7holdout.txt"` and
  `ml_threshold=0.65`. Clinical profiles unchanged.

## Validation

- **PSG-IPA reproducibility:** 10/10 tests pass in 11:59
  (clinical-profile defaults unchanged → paper v31 strict / standard
  / sensitive numerics bit-identical to v0.5.2 and earlier).
- **MESA q=7 holdout:** with the natively-loaded `mesa_shhs`
  profile and threshold 0.65,
  **bias $-0.02$/h, MAE 5.34/h, Pearson $r$ 0.872, weighted $\kappa$
  0.497, severity-match 63\%** vs the v0.5.2 rule-based baseline
  (bias $+1.10$, $r$ 0.804, $\kappa$ 0.481, sev 59\%). Threshold
  sweep (paper v35 §3.6.1) exposes the calibration curve from
  0.45 to 0.70.
- **Cross-validation:** 5-fold AUC $0.818\pm 0.006$;
  within-distribution test AUC $0.811$.

## Known limitations carried forward

- Severe-AHI under-detection on Compumedics (paper v34 §S5.7–S5.8:
  mesaid 6382 with 91% never-seen events) is upstream of this
  re-classifier — the LightGBM cannot recover events the rule-based
  detector never proposes. Closing that residual is targeted for
  v0.7 (multi-channel candidate fusion or deep-feature extraction).

# v0.5.2 — 2026-05-04

`mesa_shhs` profile: cohort-specific re-enablement of envelope smoothing
and a lower consecutive-breath threshold for peak-based detection. No
psgscoring source-code changes — purely two field overrides on the
`mesa_shhs` PostProcessingRules. Clinical profile defaults
(`aasm_v3_*`, `aasm_v2_rec`, `aasm_v1_rec`, `cms_medicare`,
`chicago_1999`) are unchanged and PSG-IPA reproducibility is preserved.

## Changed (`mesa_shhs` profile only)

- `mesa_shhs.post_processing.flow_smoothing_s = 3.0` (was 0.0).
  Re-enables the 3-second envelope smoothing that was the pre-v0.2.8
  default. v0.2.8 removed it from clinical defaults because it caused
  +54 false hypopneas on PSG-IPA SN1 (severity drift Mild → Moderate),
  but the MESA Compumedics signal characteristics differ from the
  PSG-IPA Vitalograph hardware — the smoothing bridges brief noise
  excursions in the Pres envelope and lets legitimate ≥10s
  flow-reduction periods form contiguous below-threshold runs in the
  candidate-formation step.
- `mesa_shhs.post_processing.peak_min_consecutive_breaths = 2`
  (was 3). Lowers the run-length requirement for peak-based detection
  to match the existing sensitive profile. Recovers brief
  high-amplitude-drop sequences typical of dense MESA event clusters
  that the 3-breath rule lets slip through.

## Validation

- **PSG-IPA reproducibility:** clinical profiles unchanged → expected
  10/10 pass; verified post-commit.
- **MESA q=7 n=99:** with the v0.5.2 `mesa_shhs` profile,
  bias **+1.10**/h (was −1.55 on v0.5.1, still within Anderer 2022
  stretch target ≤3),
  MAE 6.06 (≈ unchanged),
  SD 8.43 (down from 8.97),
  Pearson $r$ **0.804** (up from 0.775, +0.029),
  weighted $\kappa$ **0.481** (up from 0.400, +0.081 — into the
  DRIVEN range 0.55-0.65 lower bound),
  severity-match **59%** (up from 53%, +6 pct).
- **Mesaid 6382 specifically:** algo AHI 4.7 vs v0.5.1's 4.3 — the
  noise on this single recording is too extreme even for the
  smoothing-plus-min-breaths-2 combination; cohort-level lift comes
  from other recordings benefiting from the smoother envelope.

## Caveat

The smoothing re-enablement is justified because clinical profiles
keep `flow_smoothing_s=0.0` and so cannot regress on PSG-IPA. Any
future deployment of `mesa_shhs` for non-MESA-style cohorts should
re-evaluate whether 3.0s is still appropriate for the target sensor
characteristics.

# v0.5.1 — 2026-05-03

Profile-tunable dynamic baseline parameters and opt-in RIPsum fallback
for the hypopnea channel. Both adaptations were motivated by the
research review of best-performing tools on MESA/SHHS (Vaquerizo-Villar
DRIVEN, Nassi WaveNet, Lazazzera per-sample digitised, Koley & Dey
robust airflow envelope tracking). Defaults preserve paper v31
PSG-IPA reproducibility (10/10 pass on
`tests/test_psgipa_reproducibility.py`).

## Added

- `PostProcessingRules.baseline_window_s` (default `300`).
  Sliding-window length (seconds) for `signal.compute_dynamic_baseline`.
  Previously hard-coded.
- `PostProcessingRules.baseline_percentile` (default `95.0`).
  Envelope percentile used as the local baseline anchor in
  `compute_dynamic_baseline`. Previously hard-coded as 95.
- `PostProcessingRules.flow_fallback_strategy` (default `"none"`).
  When set to `"ripsum_on_nasal_failure"`, the pipeline checks
  the Pres-signal amplitude excursion; if median 30-second
  peak-to-trough excursion is below 0.5% of the 99th-percentile
  amplitude, OR the night-wide excursion-CV is below 0.10
  (signal flat / sensor disconnected), the thoracoabdominal RIP
  sum (thorax + abdomen) is substituted as the hypopnea
  detection input. The quality test runs in
  `pipeline._maybe_apply_ripsum_fallback`.

## Changed (`mesa_shhs` profile only)

- `mesa_shhs.post_processing.baseline_window_s = 120`
  (Lazazzera 2020-style 2-minute window; clinical profiles keep 300).
- `mesa_shhs.post_processing.baseline_percentile = 85.0`
  (Lazazzera/Koley range 80–90; clinical profiles keep 95).
- `mesa_shhs.post_processing.flow_fallback_strategy = "ripsum_on_nasal_failure"`.

## Validation

- **PSG-IPA reproducibility (regression):** 10/10 tests pass in 11:58.
  Paper v31 strict/standard/sensitive numerics on SN1–SN5 are
  bit-identical because clinical-profile defaults are unchanged.
- **MESA q=7 n=99:** with the v0.5.1 `mesa_shhs` profile,
  bias $-1.55$/h (was $-0.78$ on v0.5.0), MAE 6.05 (≈unchanged),
  Pearson $r$ 0.775 (up from 0.759, +0.016), weighted $\kappa$ 0.40
  (unchanged), severity-match 53% (≈unchanged).
- **Per-recording diagnostic on `mesaid` 6382:** the RIPsum
  fallback did NOT trigger (Pres signal `rel_exc=0.352`,
  `cv=2.808`; both well above the trigger thresholds), confirming
  that the under-detection on this recording is not driven by
  nasal-pressure quality but by something deeper in the
  airflow-envelope or breath-segmentation logic. The MESA cohort
  outliers thus remain a v0.6 (real algorithmic rework) target;
  v0.5.1 nevertheless ships these knobs because they (a) are useful
  for cohorts where the Pres signal does degrade, and (b) the
  baseline window/percentile tuning produced a small but real
  $r$ improvement on the highest-quality MESA stratum.

## Known limitations carried forward

- Severe under-detection on dense-cluster OSA recordings
  (\S~S5.7–S5.8 of paper v34) remains. The flow-detection-sensitivity
  gap on `mesaid` 6382 (91% never-seen candidates) is not closed by
  v0.5.1's profile-tunable baseline; the gap appears upstream of the
  baseline computation in the per-breath segmentation or Hilbert
  envelope construction. Identified as the v0.6 priority.

# v0.5.0 — 2026-05-03

Profile-tunable local-baseline validator and Rule 1B arousal-coupling
window. Three previously-hardcoded thresholds in `respiratory.py` are
now exposed as fields on `PostProcessingRules`, enabling per-profile
cohort-specific tuning. The MESA/SHHS dataset profile (`mesa_shhs`)
gets a corrected metadata block plus the cohort-tuned values
established in paper v34 §S5.6. Clinical profiles (`aasm_v3_*`,
`aasm_v2_rec`, `aasm_v1_rec`, `cms_medicare`, `chicago_1999`) keep the
released defaults; PSG-IPA reproducibility 10/10 pass — paper v31
numerics bit-identical.

## Added (profile-tunable; defaults preserve paper v31 reproducibility)

- `PostProcessingRules.local_baseline_min_reduction_pct`
  (default `20.0`). Floor for the local-baseline-validator's
  reduction requirement, previously the hard-coded
  `min_reduction_pct=20.0` parameter default in
  `respiratory._validate_local_reduction`.
- `PostProcessingRules.local_baseline_pre_win_s`
  (default `30.0`). Pre-event window (seconds) for the local baseline
  computation, previously hard-coded as `pre_win_s=30.0`.
- `PostProcessingRules.rule1b_arousal_window_s`
  (default `15.0`). Window within which a scored arousal must follow
  a rejected hypopnea candidate for AASM Rule 1B reinstatement,
  previously the module-level constant `RULE1B_AROUSAL_WINDOW_S=15.0`.

These three fields are wired through `constants.py:_profile_to_legacy_dict`
into the legacy SCORING_PROFILES dict, then read in
`respiratory.detect_respiratory_events` and passed down to
`_detect_hypopneas` → `_validate_local_reduction`. The Rule 1B
window is plumbed via a new `arousal_window_s` parameter on
`reinstate_rule1b_hypopneas` (kw-only; falls back to module-level
constant if `None`), and `pipeline.py` forwards
`profile["RULE1B_AROUSAL_WINDOW_S"]` from the legacy dict.

## Changed (`mesa_shhs` profile only)

- `mesa_shhs` metadata block corrected: `sensor` is now
  `"nasal_pressure_primary"` (was `"rip_bands_primary"`, which
  contradicted the MESA Sleep PSG Scoring Manual);
  `desat_threshold=0.03` (was `None` which fell back to the legacy
  default), `desat_or_arousal=True` and `desat_required=False`,
  matching the canonical NSRR `nsrr_ahi_hp3u` clinical AHI definition
  (≥30% airflow reduction with ≥3% desat OR coupled arousal).
- `mesa_shhs.post_processing` cohort-tuning per paper v34 §S5.6:
  `local_baseline_min_reduction_pct=15.0`,
  `local_baseline_pre_win_s=60.0`,
  `rule1b_arousal_window_s=5.0`.

## Validation

- **PSG-IPA reproducibility:** 10/10 tests pass (`tests/test_psgipa_reproducibility.py`).
  Paper v31 strict/standard/sensitive numerics on SN1–SN5 are
  bit-identical to v0.3.2 and v0.4.5 because clinical-profile
  `PostProcessingRules` defaults are unchanged.
- **MESA q=7 n=99 (full highest-quality stratum):** with the
  natively-tuned `mesa_shhs` profile (no monkey-patches),
  bias **$-0.78$/h**, MAE 6.04/h, SD 9.20/h, Pearson $r$ 0.76,
  weighted $\kappa$ 0.40, severity-match 54%. Within Anderer 2022
  stretch target ($|\text{bias}|<3$/h). Bit-identical to the
  monkey-patched v0.4.5 recipe documented in paper v34 §S5.6,
  confirming clean source-level integration.

## Known limitations carried forward

- The flow-detection-sensitivity gap on severe MESA recordings
  (\S~S5.7–S5.8 of paper v34: `mesaid` 6139 with 40% never-seen
  candidates, `mesaid` 6382 with 91%) is upstream of these
  profile-tunable knobs and remains a v0.5.x/v0.6 algorithmic
  rework target.

# v0.4.5 — 2026-05-02

External-arousal injection for dataset-faithful validation. Motivated
by MESA cohort smoke runs where comparison to NSRR ``ahi_hp3u``
(3%-desat OR arousal) was systematically biased low (−9.4/h on n=30,
q≥5 cohort) because the algorithm's Rule 1B reinstatement was firing
against EEG-detected arousals (or, on quiet recordings, no arousals
at all) instead of the scorer-determined arousals embedded in the
NSRR XML. Default behaviour unchanged; PSG-IPA reproducibility 10/10
pass.

## Added

- ``run_pneumo_analysis(arousal_events=...)`` — optional list of
  pre-scored arousals to use for Rule 1B reinstatement, bypassing
  internal EEG-based detection. Each item must be a dict with at
  least ``onset_s`` and ``duration_s``. Use this for dataset-faithful
  validation against scorer-derived AHI variants that include
  arousal-coupled hypopneas. When ``None`` (default), the existing
  EEG-based path runs as before.

## Fixed

- ``pipeline.py`` — removed a duplicate local import of
  ``_compute_summary`` inside ``run_pneumo_analysis`` that shadowed
  the module-level binding and would have raised
  ``UnboundLocalError`` whenever Rule 1B reinstatement actually fired
  on a recording. The bug was latent because PSG-IPA's EEG-detected
  arousal stream rarely produces reinstatable hypopneas; on MESA with
  injected arousals the path lights up and the shadowing surfaced.

## Validation context (informational, not part of the release)

- MESA n=30, q≥5, seed=42, profile ``aasm_v2_rec`` + injected NSRR
  arousals: bias **−2.52/h** (within Anderer 2022 reasonable target
  ≤5), MAE 7.46, weighted κ 0.241, severity-match 51.7%. Same cohort
  with no arousals: bias −9.43, κ 0.045, severity-match 44.8%.
- A diagnostic on mesaid 6139 (severe AHI, ref 42.4/h → algo 10.9/h
  default): 406/411 hypopnea candidates rejected by
  ``_validate_local_reduction``. Even with the profile-tunable
  stability branch fully relaxed, only ~16 events recoverable due to
  a hardcoded ``min_reduction_pct=20.0`` floor that is **not**
  profile-aware. Making it profile-tunable is queued for v0.5; the
  larger gap (~198 events the algorithm never sees as candidates) is
  a flow-detection-sensitivity issue specific to MESA Compumedics
  data and out of scope for v0.4.x.

# v0.4.4 — 2026-05-01

Algorithm-review release. An internal v0.4.x review flagged 8
behavioural concerns and 3 AASM-mapping documentation gaps. After
PSG-IPA cross-validation, the changes that materially affected the
paper v31 numerics were demoted from default-changing fixes to
**documented opt-in parameters** so default behaviour is unchanged
and paper v31 reproduction passes. The genuinely-defensive fixes
(B1, B5, B8) and the documentation gaps are applied as defaults.

## Fixed (default behaviour change)

- **B1** `respiratory.py:_validate_local_reduction()` — when <3 s of
  pre-event signal is available, the validator was returning
  ``(True, 100.0)`` and downstream consumers treated this sentinel as
  a real measurement. Now returns ``(True, float('nan'))`` so the
  "not measured" case is unambiguous. Same for flat-line baselines.
- **B5** `ecg_effort.ecg_effort_assessment()` — added an
  ``evidence_strength`` field (``'dual'`` vs ``'spectral_only'``) so
  the upstream confidence penalty in `classify.py:Rule 5b` is
  inspectable. The penalty itself was already correctly applied
  (Rule 5b uses 0.75 vs 0.85 based on ``ecg_effort_present``).
- **B8** `plm._detect_lm_channel()` — the ``unit='auto'`` heuristic
  was 1000× wrong for mV-scaled EDF data. Replaced with a three-band
  heuristic (V → ×1e6, mV → ×1e3, µV → no scaling) and added an
  explicit ``leg_unit`` parameter so callers can pass the EDF
  physical unit rather than relying on amplitude inference. The 8 µV
  AASM amplitude threshold is unit-sensitive.

## Added (opt-in parameters; defaults preserve paper v31 numerics)

- **B7** `spo2.get_desaturation()` gained
  ``global_baseline_min_local_pct`` (default ``None`` = paper v31
  always-override behaviour). Set to a value (e.g. 88) to gate the
  global-baseline override so it only fires when the local baseline
  is implausibly low. Helps avoid artificially inflating the baseline
  for chronic-desaturator patients (COPD, OHS). The
  ``early_nadir_min_drop_pct`` default stays at 5.0 (paper v31);
  pass 3.0 to align with the AASM ≥3% criterion.

## Documented (no behaviour change)

- **B2** `_validate_local_reduction()` — full docstring rewrite with
  explicit AASM-mapping note, paper reference, and instructions for
  disabling the stability-aware tightening per profile (set
  ``stability_strict_reduction == min_reduction_pct``).
- **B3** `classify.py:Rule 6` — comment added documenting the
  deliberate AASM-deviation (effort 0.30–0.40 defaults to central,
  not obstructive) introduced in v0.8.30 to handle cardiac-pulsation
  artefact. Default unchanged; lower the 0.40 threshold to 0.30 in a
  fork to revert to AASM-strict behaviour.
- **B4** `signal_quality.py:FALLBACK_OBSTRUCTIVE_RATIO` — comment
  added flagging that single-channel-fallback may misclassify
  cardiac-pulsation-only events as obstructive at the 0.50
  threshold. A 0.70 threshold would be more conservative; default
  unchanged for backward compatibility.
- **B6** `ancillary.detect_cheyne_stokes()` autocorrelation peak
  threshold — docstring NOTE added pointing out that the literature
  uses tighter thresholds (Trinder *Sleep* 1991: >0.4; He et al.
  *EHJ* 2023: >0.5). Default kept at 0.3 for paper v31 compatibility;
  v0.5 will expose the threshold as a parameter.
- **G1** `classify.py:Rule 5b` — added comment noting that
  pattern-level CSR reclassification (the AASM v3 ≥3-consecutive-
  central + crescendo-decrescendo + ≥40 s rule) is detected by
  `ancillary.detect_cheyne_stokes()` and applied downstream by
  `postprocess.reclassify_csr_events()`, not in classify.py.
- **G2** `postprocess.decompose_mixed_apneas()` — documented the
  AASM-conform "leading low-effort = central phase" assumption.
- **G3** `postprocess.reclassify_csr_events()` — documented that the
  preserved ``original_type`` field provides v0.4.4-interim audit-
  trail rollback for false-positive CSR reclassifications. Full
  append-only audit log is on the v0.5 roadmap.

## Reproducibility

PSG-IPA standard-profile aggregate (n=5) under v0.4.4 reproduces
paper v31 metrics bit-identically (default parameters). The
``test_psgipa_reproducibility.py`` integration tests pass with
``PSGIPA_DATA_DIR`` set.

---

# v0.4.3 — 2026-05-01

## Added

- **`tests/test_psgipa_reproducibility.py`** — pytest that asserts
  paper v31 metrics on PSG-IPA (bias, MAE, Pearson r, F1 SN3, mean Δt)
  within tolerance. Skipped when `PSGIPA_DATA_DIR` is not set, so CI
  without the dataset still passes; gates against silent algorithmic
  drift on full runs.
- **Robustness-grade output** in `validate_psgipa.py` (per-recording
  A/B/C grade computed across the three clinical profiles).

## Changed

- **`validate_psgipa.py` fully rewritten.** The previous v3 script
  read scorer-1 stages from `Sleep_stages/Annotations/manual/` and
  events from `Resp_events/Annotations/manual/` and applied a
  `meas_date` shift to align them; that cross-subtree alignment
  introduced small epoch-attribution errors and produced
  bias +3.6/h on PSG-IPA instead of paper v31's +1.8/h.
  The new harness is faithful to paper v31 supplement S3.2: scorer-1
  stages and events are both read from
  `Resp_events/Annotations/manual/{SN}_Respiration_manual_scorer1.edf`,
  which shares its time axis with the primary Respiration EDF.
  Reproduces paper v31 standard-profile metrics bit-identically and
  emits a robustness grade per recording.
- The harness now runs all three clinical profiles
  (`aasm_v3_strict`, `aasm_v3_rec`, `aasm_v3_sensitive`) per recording
  and emits a JSON payload consumed by `validation_report.py`.

## Reproducibility

- PSG-IPA standard-profile aggregate (n=5):
  bias +1.77/h, MAE 1.84/h, SD 2.00/h, LoA [-2.15, +5.68]/h,
  Pearson r 0.997, weighted κ 0.84, F1 SN3 0.886, mean Δt SN3 1.97 s.
  Standard-profile per-recording AHIs reproduce paper v31 Table 1
  bit-identically; strict and sensitive per-recording AHIs differ
  from v31 because v0.4.0 retuned those profile parameters (see
  the v0.4.0 entry below).

---

# v0.4.2 — 2026-04-29

## Fixed
- **Local baseline validation now profile-aware.** The hardcoded
  `local_cv < 0.30` stability check in `_validate_local_reduction`
  is now driven by two new profile parameters
  (`local_baseline_cv_threshold` and `local_baseline_strict_reduction`).
- **Scope bug fix:** `sp` dict was incorrectly referenced inside
  `_detect_hypopneas` (which doesn't receive `sp`); the new
  parameters are threaded through the call chain via function
  arguments.

## Added
- `PostProcessingRules.local_baseline_cv_threshold` (default 0.30)
- `PostProcessingRules.local_baseline_strict_reduction` (default 25.0)
- Per-profile values:
    - `aasm_v3_strict`: cv=0.30, strict_reduction=30.0
    - `aasm_v3_rec`: cv=0.30, strict_reduction=25.0
    - `aasm_v3_sensitive`: cv=0.20, strict_reduction=20.0
    - 5 other profiles: defaults

## Changed
- `_detect_hypopneas` and `_validate_local_reduction` signatures
  extended (backward compatible — defaults match legacy behaviour)
- `_profile_to_legacy_dict` exports new `LOCAL_BL_CV_THRESHOLD` and
  `LOCAL_BL_STRICT_RED` keys

## Validation
- PSG-IPA aggregate metrics improved: r=0.994, kappa=0.800,
  F1 SN3=0.860, mean Δt=1.39s
- Severity concordance 4/5 (paper v31 claim retained)
- Profile-sweep monotonie not yet fully restored on borderline
  cases (SN2, SN4); to be addressed in future release after
  deeper review of flow_smoothing × peak_detection × local_baseline
  interaction

---

# v0.4.1 — 2026-04-27

## Fixed
- **Profile parameter integration bug** (cause of monotonie violations
  in v0.4.0). The hardcoded stability filter threshold `0.45` in
  `respiratory.py` ignored the per-profile `stability_filter_cv` and
  `peak_min_consecutive_breaths` parameters. Now correctly read from
  scoring profile dict, restoring intended profile differentiation.
  PSG-IPA validation re-run shows expected monotonic ordering
  (strict ≤ standard ≤ sensitive for hypopnea-dominated recordings).

## Added
- **3-point clinically calibrated confidence sweep** in scoring summary:
    - `oahi_sweep`: `{lenient: c≥0.30, primary: c≥0.47, strict: c≥0.65}`
    - `oahi_sweep_width`: max−min spread in /h
    - `robustness_grade`: 'A' (<5/h), 'B' (5-10/h), 'C' (≥10/h)
  Calibrated to AASM inter-scorer variability (~10-20% AHI).
  Mean sweep width on PSG-IPA: 3.9/h. Replaces use of legacy 4-point
  `oahi_thresholds` for clinical interpretation; the latter is
  preserved for backward compatibility.
- New TypedDict `OAHISweep3pt` for IDE/mypy support.

## Changed
- `oahi_thresholds` (legacy 4-point sweep) is no longer used in
  clinical UI displays but kept in output for compatibility.
- Profile parameters `STABILITY_FILTER_CV` and
  `PEAK_MIN_CONSECUTIVE_BREATHS` are now properly threaded through
  to the detection pipeline.

## Notes
- The monotonie-fix is a behavioural change: profile differentiation
  now produces meaningfully different OAHI values. Re-validation
  on PSG-IPA recommended for users comparing to v0.4.0 results.
- All 8 historical profiles (aasm_v3_rec/strict/sensitive,
  aasm_v2_rec, aasm_v1_rec, cms_medicare, mesa_shhs, chicago_1999)
  remain available.

---

# v0.4.0 — 2026-04-26

Major refactor: introduces the unified scoring-profile framework.
Eight named profiles ship in `psgscoring.PROFILES`, exposed via the
`scoring_profile=` kwarg of `run_pneumo_analysis()`.

## Added

- **Three clinical profiles** (`PROFILE_GROUPS["clinical"]`):
  - `aasm_v3_strict`   — conservative variant of AASM v3 Rule 1A
  - `aasm_v3_rec`      — recommended (3%-or-arousal hypopnea)
  - `aasm_v3_sensitive` — UARS-oriented sensitive variant
- **Five historical / dataset profiles**:
  `aasm_v2_rec`, `aasm_v1_rec`, `cms_medicare`, `mesa_shhs`, `chicago_1999`.
- Profile dataclasses: `HypopneaRules`, `ApneaRules`, `SpO2Rules`,
  `PostProcessingRules`, aggregated by `Profile`.
- `PROFILE_GROUPS`: convenience aliases (`clinical`, `aasm_era`,
  `coverage`, `dataset`, `full_6`, `all`).
- Legacy aliases `strict` / `standard` / `sensitive` accepted for
  backward compatibility (deprecation-warning, removal planned in v0.5.0).

## Changed — strict vs. v0.3.x defaults

The strict profile is a **deliberate tightening** relative to the
single hardcoded default of v0.3.x:

- `stability_filter_cv` 0.30 (no longer hardcoded 0.45)
- `breath_level_detection` off  ·  `flow_smoothing_s` 0
- `spo2.nadir_search_s` 30 s  ·  `local_baseline_strict_reduction` 30

## Changed — sensitive vs. v0.3.x defaults

The sensitive profile is a **deliberate loosening** for UARS detection:

- `hypopnea.flow_reduction_threshold` 0.25 (vs. 0.30 in rec/strict)
- `hypopnea.max_duration_s` 90 s (vs. 60 s in rec/strict)
- `flow_smoothing_s` 5.0  ·  `breath_level_detection` on
- `peak_min_consecutive_breaths` 2 (vs. 3 in rec/strict)
- `stability_filter_cv` 0.50  ·  `local_baseline_cv_threshold` 0.20
- `local_baseline_strict_reduction` 20.0
- `artefact_flank_exclusion` off (deliberate; reduces false negatives
  on flow recovery slopes)
- `apnea.max_duration_s` 120 s (vs. 90 s in rec/strict)

## Reproducibility note vs. paper v31

Paper v31 (Rombaut et al. 2026) was generated against
**psgscoring v0.3.2**, where strict / standard / sensitive were
configurable presets sharing a single hardcoded stability-filter
threshold. The v0.4.0 profile system makes that threshold (and
several other rules) profile-specific, which causes the strict and
sensitive per-recording AHIs on PSG-IPA SN1-SN5 to diverge from the
v31 Table 1 values. The standard (`aasm_v3_rec`) profile remains
parameter-equivalent for the rules active on this dataset and
reproduces v31's standard-profile AHIs and aggregate metrics
(bias +1.8/h, MAE 1.8/h, r 0.997, F1 SN3 0.886) bit-identically.
Users wishing to reproduce v31 Table 1 verbatim should pin to v0.3.2:
`pip install psgscoring==0.3.2`.

## Known issue (fixed in v0.4.1)

The profile parameters introduced in v0.4.0 were not actually read
by `respiratory.py`, which kept a hardcoded 0.45 stability-filter
threshold. PSG-IPA monotonie was therefore not yet established in
v0.4.0; the v0.4.1 release wires the parameters through.

---

# v0.3.2 — 2026-04-21

Bugfix release.

## Fixed
- `signal_quality_channels._check_montage()`: numpy boolean-ambiguity error
  caused by `flow = _get("flow") or _get("flow_pressure")` when both channels
  returned ndarrays. Replaced with explicit `None`-check. This bug was
  silently caught by the pipeline exception handler, leaving
  `output["channel_quality"]` populated with `{"overall_grade": "unknown"}`,
  and was a no-op for classification (which reads the separate
  `output["signal_quality"]` from `compare_rip_pair`), but prevented
  channel-level quality metadata from reaching PDF reports and validation
  exports. Detected during PSG-IPA re-validation on 2026-04-21.

# Changelog

## v0.3.1 — 2026-04-21

### Added

- `classify_apnea_type()` accepts optional `signal_quality` parameter
  (output of `compare_rip_pair`). When gate reports
  `recommended_mode="single-channel"`, classification routes to
  `single_channel_fallback_classify()` on the working channel instead
  of the bilateral 7-rule chain. When `"unreliable"`, returns
  `"uncertain"`.

### Fixed

- **Bug 2: respiratory classifier now consumes the RIP-pair quality
  gate.** The v0.2.962 gate detected single-sensor failures correctly,
  but `classify_apnea_type` ignored the signal, defaulting to
  obstructive classification for dead-sensor events. This caused the
  Loos case (AZORG April 2026) to appear as 100% obstructive while the
  true underlying pathology was likely CSAS (CAI 45.1 on
  abdomen-only analysis).

### Changed

- Classifier decision chain has new Rule -1 (RIP pair quality gate)
  before Rule 0 (phase angle). With `signal_quality=None` or
  `recommended_mode="bilateral"`, behavior is unchanged from v0.2.963.
- Pipeline reorders signal_quality computation to run BEFORE respiratory
  event detection (previously was after, as a documentation-only step).

### Clinical impact

Patients with single RIP sensor failures now receive classification
based on the working channel instead of a mechanical obstructive
default. The clinically important shift is for central sleep apnea
syndrome (CSAS) detection in patients whose thorax or abdomen RIP
sensor fails during the recording.

### Test coverage

- `tests/test_bug2_classifier_quality_gate.py`: 9 passing tests
  covering fallback in isolation, end-to-end classifier routing,
  bilateral preservation, and unreliable-gate scenarios.

---



## v0.2.963 — 2026-04-20

### Fixed

- **assess_rip_channel 2D input regression** — MNE's `raw.get_data(picks=[ch])`
  returns shape `(1, N)` even for single-channel requests. The welch()
  PSD computation on a 2D input produced a 2D output, which then broke
  1D boolean masking in the breath-band energy calculation. The fix
  squeezes the input to 1D at the top of `assess_rip_channel()` and
  returns a defensive 'failed' status for genuinely higher-dimensional
  input.

### Clinical impact

  Without this fix, signal quality assessment silently failed in the
  real deployment pipeline, leaving the RIP pair quality gate
  ineffective at detecting single-sensor failures. The Loos case
  (AZORG April 2026, thorax RIP dead, ratio 6862×) was the motivating
  clinical scenario.

### Added

- `tests/test_signal_quality_2d.py` — 5 regression tests covering
  1D baseline, 2D MNE-shape input, higher-dimensional defensive
  rejection, and Loos-like single-sensor failure scenarios.

---



All notable changes to **psgscoring** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.951] 

### Added

- **Ensemble-averaged hypoxic burden** (`baseline_method="ensemble"`).
  Subject-specific search window derived from the ensemble average of all
  time-aligned SpO₂ curves, reproducing the original Azarbarzin et al.
  (Eur Heart J 2019) method.
- Pre-event baseline = SpO₂ at left peak of ensemble curve; area
  integrated within the ensemble-derived search window.
- Automatic fallback to percentile method when fewer than 3 events available.
- Helper function `_ensemble_search_window()` for ensemble curve computation.

### Changed

- `compute_hypoxic_burden()` now accepts a `baseline_method` parameter:
  `"percentile"` (default, backward compatible) or `"ensemble"`.
- Return dict now includes `baseline_method` and `ensemble_window_s` keys.
- `spo2.py`: 395 → 545 lines (+150).

### References

- Azarbarzin A et al. The hypoxic burden of sleep apnoea predicts
  cardiovascular disease-related mortality. *Eur Heart J*.
  2019;40(14):1149-1157.
- He S, Cistulli PA, de Chazal P. Comparison of oximetry event
  desaturation transient area-based methods. *IEEE EMBC*. 2024.

---

## [0.2.95]

### Changed

- Maintenance release consolidating v0.2.93 and v0.2.94 fixes prior to
  the ensemble-HB addition in v0.2.951.
- Internal cleanup of post-processing return dictionary keys for
  consistency across CSR, mixed decomposition, and CII outputs.

---

## [0.2.94] 

### Fixed

- Stale documentation references to v0.2.92 updated to v0.2.93 in
  `DISCLAIMER.md` and module-level docstrings.
- Minor correction to CSR cycle-matching tolerance (±2 s) in
  `postprocess.reclassify_csr_events()` to avoid over-reclassification
  on recordings with irregular CSR periodicity.

---

## [0.2.93]

### Added

- **iSLEEPS validation** (Maiti et al., *Sci Data* 2026, 39 ischemic
  stroke patients). Mean absolute error 3.3 /h for normal/mild severity;
  systematic under-scoring at moderate/severe consistent with the high
  prevalence of central and mixed apneas in stroke populations.
- **Event-level temporal validation** on PSG-IPA severe-OSA recording
  (SN3, 322 reference events): F1 = 0.890, IoU = 0.866, mean onset
  difference Δt = 2.3 s (IoU ≥ 0.20 matching).

### Changed

- CSR reclassification threshold tuned from v0.2.92 pilot runs on
  iSLEEPS (365 events reclassified across 36/96 studies).

---

## [0.2.92] 

### Added

- **Hypoxic burden** (`spo2.compute_hypoxic_burden`).
  Total area of SpO₂ desaturation associated with respiratory events,
  normalised per hour of sleep (%·min/h), following Azarbarzin et al.
  (AJRCCM 2019).
  - Per-event integration from onset to SpO₂ recovery.
  - Pre-event baseline (90th percentile, 120 s window) with global
    95th-percentile fallback.
  - Clinical thresholds: <20 low, 20–73 moderate, >73 high CV risk.
  - Automatically computed in `run_pneumo_analysis()` (Step 10).
  - Available at `output["hypoxic_burden"]` and
    `output["spo2"]["summary"]["hypoxic_burden"]`.
  - NumPy ≥2.0 compatible (`np.trapezoid` with `np.trapz` fallback).
- **Post-processing module** (`postprocess.py` — new).
  - `reclassify_csr_events()`: CSR-flagged obstructive/mixed events
    reclassified as central (addresses cardiac pulsation artefact in
    heart failure).
  - `decompose_mixed_apneas()`: analyses effort signal to measure
    central vs. obstructive portion; reclassifies to central if the
    central portion is ≥10 s.
  - `compute_central_instability_index()`: quantifies
    profile-dependent uncertainty in obstructive/central
    classification on a 0–1 scale.
  - `postprocess_respiratory_events()`: master function calling all
    three.
  - Automatically runs in `run_pneumo_analysis()` (Step 11).
  - Results at `output["postprocess"]`.

### Changed

- Pipeline now has 11 steps (was 9): Step 10 added hypoxic burden
  computation; Step 11 added post-processing.
- Public API: 42 exports (was 38). New: `compute_hypoxic_burden`,
  `postprocess_respiratory_events`, `reclassify_csr_events`,
  `decompose_mixed_apneas`, `compute_central_instability_index`.

### References

- Azarbarzin A et al. The hypoxic burden of sleep apnoea is an
  independent predictor of incident cardiovascular outcomes.
  *AJRCCM*. 2019;200(2):211-219.

---

## [0.2.91] 

### Added

- **External validation on PSG-IPA** (PhysioNet, Bakker et al.
  *Physiol Meas* 2021): 5 recordings, 59 scorer sessions from up to
  12 certified RPSGT/ESRS technologists.
  Mean AHI bias +1.6 /h, mean |ΔAHI| = 1.8 /h, Pearson r = 0.990,
  AASM severity concordance 4/5 (80 %).
- **Stability-aware threshold** (Fix 6 refinement): when local breath
  coefficient of variation is <0.30, the 70 % hypopnea threshold is
  tightened to 30 % reduction to avoid over-counting in stable
  breathing.
- **Consecutive breath requirement** (Fix 7 refinement): peak-based
  hypopnea detection now requires ≥3 consecutive sub-threshold
  breaths before flagging, reducing sensitivity to single aberrant
  breaths.

### Changed

- Documentation: added "AHI confidence interval" as a named feature
  in README and PyPI project description.
- DUA-ready project description updated for MESA/SHHS data-access
  applications (primary dataset MESA due to dual-sensor support;
  SHHS secondary for thermistor-only validation).

---

## [0.2.9]

### Added

- **Dual-sensor flow detection** per AASM.
  Apneas are now scored on the oronasal thermistor signal; hypopneas
  on the nasal pressure transducer, following AASM recommendations.
  - Channel auto-detection via transducer metadata and channel-name
    patterns.
  - Intelligent fallback: when only one flow channel is available, it
    is used for both event types (backward compatible).
  - Result metadata (`meta.flow_channels`) logs which sensor was used
    for which event type.

### Fixed

- Channel-name matching is now order-independent (earlier versions
  picked the first match, which could cause thermistor
  misclassification on devices that list both channels generically).

---

## [0.2.8] 

### Added

- **AHI confidence interval** with robustness grading.
  Every analysis now runs three scoring profiles simultaneously and
  reports the AHI as an interval rather than a point estimate.
  - **Profiles**: `strict`, `standard` (default), `sensitive`.
  - **Robustness grade**: `A` (all three profiles agree on severity
    — treatment decision unambiguous), `B` (two of three concordant
    — probable, clinical correlation recommended), `C` (all discordant
    — manual review recommended).
  - Output at `results["ahi_interval"]` with fields `strict`,
    `standard`, `sensitive`, and `robustness_grade`.
- **Breath-amplitude stability filter** (Fix 6).
  For each hypopnea candidate, the coefficient of variation of breath
  amplitudes in the surrounding four minutes is computed. Candidates
  with CV < 0.45 (stable, non-pathological breathing) are rejected
  as normal variability rather than true events.
  - Ablation on PSG-IPA SN4 (normal OSA): 56 false-positive hypopneas
    rejected, correcting Mild → Normal.
  - Ablation on PSG-IPA SN3 (severe OSA): 11 events rejected (−3 %),
    confirming the filter targets false positives rather than true
    pathology.

### Removed

- **3-second flow smoothing** removed from the standard profile.
  Ablation analysis on PSG-IPA SN1 identified 3-second smoothing as
  the dominant source of over-counting, bridging recovery breaths into
  continuous "reduced flow" segments and generating +54 false
  hypopneas on a single mild-OSA recording. The smoothing shifted
  severity classification from Mild to Moderate.
  - SN1 standard-profile AHI: 15.9 → 8.1 (−49 %).
  - SN2 standard-profile AHI: 21.9 → 9.3 (−57 %).
  - SN4 standard-profile AHI: 13.6 → 4.3 (−68 %).
  - SN3 (severe) standard-profile AHI: 55.6 → 53.8 (−3 %,
    true events preserved).
- `HYPOPNEA_SMOOTH_S` constant deprecated (now 0.0 for standard).

### Changed

- Scoring profiles module reorganised into a central
  `constants.SCORING_PROFILES` dictionary.
- `run_pneumo_analysis()` now accepts a `scoring_profile` parameter
  (default `"standard"`).
- Internal event-detection pipeline calls all three profiles in
  sequence and combines results into the interval/grade structure.

---

## [0.2.7] 

### Changed

- Stability release of the TECG and spectral-effort modules
  introduced in v0.2.4. Reference version cited in manuscript v25.
- Minor performance improvement in R-peak detection for noisy ECG:
  fallback to Pan-Tompkins when WFDB method fails.

### Fixed

- `ecg_effort.ecg_effort_assessment()` previously returned `None` when
  the ECG channel was short (<30 s); now returns a dict with
  `assessment="insufficient_data"` and no reclassification is applied.

---

## [0.2.6] 

### Fixed

- Edge case in `classify.classify_apnea_type()` where ECG-based Rule
  5b reclassification could conflict with Rule 0 (Hilbert
  phase-angle), producing inconsistent labels. Rule precedence now
  documented: 0 → 5 → 5b, with 5b only applied when Rule 5 produced
  an "uncertain" result.
- Minor numerical stability fix in Hilbert phase-angle computation
  for recordings with intermittent RIP channel dropouts.

---

## [0.2.4] 

### Added

- **ECG-derived effort classification** (new module
  `psgscoring/ecg_effort.py`).
  - **Transformed ECG (TECG)** method (Berry et al., *JCSM* 2019):
    QRS blanking + 30 Hz high-pass filtering to reveal inspiratory
    EMG bursts from intercostal muscles.
  - **Spectral effort classifier**: cardiac (0.8–2.5 Hz) vs.
    respiratory (0.1–0.5 Hz) power analysis on RIP bands during
    apnea events; flags cardiac dominance.
  - **Combined reclassification logic**: events reclassified as
    central when *both* TECG (no inspiratory bursts) *and* spectral
    analysis (cardiac spectral dominance) agree.
  - New output field `n_ecg_reclassified_central` in respiratory
    results.
- Public API: `ecg_effort_assessment`, `compute_tecg`,
  `detect_r_peaks`, `qrs_blanking`, `detect_inspiratory_bursts`,
  `spectral_effort_classifier`.

### Changed

- `pipeline.py`: ECG channel now extracted and passed to respiratory
  scoring when available; graceful degradation when absent.
- `respiratory.py`: TECG computed once per recording; the ECG
  assessment is passed to both apnea and hypopnea
  `classify_apnea_type()` calls.
- `classify.py`: ECG-based reclassification integrated as Rule 5b in
  the 7-rule decision tree.

### References

- Berry RB et al. Use of a transformed ECG signal to detect
  respiratory effort during apnea. *JCSM*. 2019;15(11):1653-1660.

---

## [0.2.3] 

### Notes

- Content-identical re-release of v0.2.2 for PyPI. The v0.2.2 package
  name was temporarily unavailable on PyPI at the time of publication;
  v0.2.3 was published with the same source to guarantee PyPI
  availability and avoid long-term ambiguity.

### Added

- `signal_quality.py` module: per-channel flat-line, clipping,
  disconnect, line-noise, and montage-plausibility checks.
- **Flattening-based RERA** detection (Hosselet et al. *AJRCCM*
  1998): sequences of ≥3 consecutive breaths with flattening index
  >0.30 spanning ≥10 s, terminated by an arousal.
- Dual-source RDI computation: `RDI = AHI + (FRI-RERA + flattening-RERA) / TST`.
- Hypopnea subtype counts in summary output: `n_hypopnea_obstr`,
  `n_hypopnea_central`, `n_hypopnea_mixed`.
- `assess_signal_quality()` added to public exports.

### Changed

- Pipeline now 11 steps (was 10): Step 1b added for signal-quality
  assessment, executed before event detection.

### References

- Hosselet J et al. Detection of flow limitation with a nasal
  cannula/pressure transducer system. *AJRCCM*. 1998;157(5):1461-1467.

---

## [0.2.2] 

### Added

- Initial implementation of signal-quality assessment and flattening
  RERA (see v0.2.3 notes for the content description; this version
  was tagged on GitHub but its PyPI publication was delayed to
  v0.2.3).

---

## [0.2.0] 

### Added

Initial public release of `psgscoring` as a standalone library,
extracted from the YASAFlaskified clinical platform.

**Core algorithms** (pure scipy / NumPy, no deep learning, no GPU):

- Square-root linearisation of nasal pressure (Bernoulli correction,
  Thurnheer et al. *AJRCCM* 2001).
- Dynamic 5-minute rolling baseline (95th percentile) with
  stage-specific blending.
- Hilbert envelope for instantaneous amplitude.
- MMSD artefact validation (Lee et al. 2008, κ = 0.78).
- Dual-sensor detection with exclusion masking.
- Temporally constrained SpO₂ coupling (Uddin et al. 2021).
- Two-pass Rule 1B hypopnea detection with breath-cycle validation.
- 7-rule apnea type classification (obstructive / central / mixed)
  with Hilbert phase-angle analysis.
- Two-phase arousal detection with spindle exclusion and
  cardiovascular reactivity (CVR) coupling.
- PLM scoring per AASM and WASM criteria (Zucconi 2006).
- Cheyne-Stokes detection via autocorrelation.

**Package**:

- 21 unit tests, example script, BSD-3 licence.
- GitHub Actions CI (Python 3.9 – 3.12).

### References

- Thurnheer R, Xie X, Bloch KE. Accuracy of nasal cannula pressure
  recordings. *AJRCCM*. 2001;164(10):1914-1919.
- Lee H et al. Detection of apneic events from single-channel nasal
  airflow. *Physiol Meas*. 2008;29:N37-N45.
- Uddin A et al. Automated detection of respiratory events during
  sleep from pulse oximetry and airflow signals. *Sleep Breath*.
  2021;25(1):127-138.
- Vallat R, Walker MP. An open-source, high-performance tool for
  automated sleep staging. *eLife*. 2021;10:e70092.

---

## Versions not on PyPI

The following version numbers appear in internal development history
but were never published as separate PyPI artefacts:

- **0.1.x** — pre-release development, internal only.
- **0.2.1** — skipped; improvements rolled into v0.2.2.
- **0.2.5** — skipped; improvements rolled into v0.2.6.
