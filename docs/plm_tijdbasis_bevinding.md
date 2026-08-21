# De PLM-detector rekende met de verkeerde klok

*21 augustus 2026. Gevonden tijdens het onderzoek naar de arousaldetector,
omdat beide modules hetzelfde symptoom vertoonden: het juiste AANTAL events
van de juiste DUUR, op de verkeerde MOMENTEN.*

## De fout

`psgscoring/plm.py`, `_detect_lm_channel`:

```python
win = max(1, int(sf * 0.1))                     # aantal STALEN per venster
n_w = len(filt) // win
rms = np.array([... for i in range(n_w)])       # één RMS per venster
...
dur_s = len(idx) * 0.1                          # <-- neemt 0,1 s aan
"onset_s": idx[0] * 0.1,                        # <-- idem
```

`win` is een geheel aantal stalen. Bij 256 Hz is `int(256 × 0,1) = 25`, en
25 stalen duren **0,09766 s**, niet 0,1 s. De omzetting van vensterindex naar
tijd vermenigvuldigt toch met 0,1.

De fout is niet constant maar **stapelt lineair over de nacht**:

| opname | sf | venster | drift aan het eind |
|---|---:|---:|---:|
| PSG-IPA SN1 (7,4 u) | 256 Hz | 0,09766 s | **+620 s** (10,3 min) |
| PSG-IPA SN3 (7,8 u) | 256 Hz | 0,09766 s | +657 s (10,9 min) |
| MESA (12 u) | 256 Hz | 0,09766 s | +1013 s (16,9 min) |

De richting is altijd dezelfde: events worden **te laat** gerapporteerd.

Het treft de gangbare sample rates 256 Hz (2,3 % fout) en 128 Hz (6,3 %
fout: `int(12,8) = 12`). Bij 100, 200 en 500 Hz is `sf × 0,1` toevallig
geheel en is er geen fout — wat verklaart waarom dit nooit is opgevallen.

## Wat het kostte

Event-F1 tegen twaalf menselijke scoorders op PSG-IPA, greedy IoU-matching
op 0,20, linkerbeen tegen `limb movement@@emg lat`:

| | huidig | tijdbasis gecorrigeerd | mens tegen mens |
|---|---:|---:|---:|
| SN1 | 0,038 (431 ev) | **0,592** (372 ev) | 0,745 |
| SN2 | 0,014 (152 ev) | **0,699** (137 ev) | 0,820 |
| SN3 | 0,045 (341 ev) | **0,692** (278 ev) | 0,909 |

De detector was dus niet slecht in wat hij vindt — hij zette het op de
verkeerde tijd. Met de klok recht komt hij op ~0,15 van de menselijke
overeenstemming.

## Wat er verderop van afhangt

De drift blijft niet binnen de PLM-module. `pipeline.py:1968-1979` berekent
`plm_arousal_index` door een PLM-onset te koppelen aan een arousal-onset
binnen **−0,5 tot +3 s**. Met onsets die door de nacht tot tien minuten
verschuiven is die koppeling toeval, en `plm_arousal_index` staat in het
klinische PDF-rapport (`generate_pdf_report.py:2351`). Hetzelfde geldt voor
`_exclude_resp_associated`, dat beenbewegingen uitsluit die binnen enkele
seconden na een respiratoir event vallen: die uitsluiting werkt nu op
verschoven tijden.

Twee dingen die daar bovenop komen en apart aandacht verdienen:

- `result["events"]` is `plm_eligible[:200]` — alleen bewegingen in slaap
  zonder respiratoire koppeling, afgekapt op 200. Op SN1 zijn dat er 200 van
  660 gedetecteerde bewegingen. Alles wat verderop `output["plm"]["events"]`
  leest — inclusief `plm_arousal_index` en de signaalweergave — ziet dus
  hooguit de eerste 200 van de nacht.
- YASAFlaskified schrijft deze onsets door naar de **EDF+-export**
  (`generate_edfplus.py:155-161`). Wie dat bestand in een viewer opent, ziet
  de PLM-markering tot tien minuten naast de beweging staan. Dat is de
  afleverkant: de bibliotheek kan het getal intern consistent hebben, maar
  wat de gebruiker ziet is verschoven.
- De menselijke PSG-IPA-annotaties staan PER BEEN
  (`limb movement@@emg lat` en `@@emg rat`), terwijl de module bilateraal
  samenvoegt binnen 0,5 s. Voor de validatie hierboven is daarom per been
  vergeleken.

## Waarom dit niet eerder opviel

Het AANTAL klopte al. `n_lm_total` en de PLM-index worden per uur berekend en
zijn ongevoelig voor een verschuiving; alleen een event-voor-event
vergelijking legt het bloot. Die vergelijking bestond niet: de PLM-module is
tot 20 augustus 2026 nooit tegen menselijke scoorders gelegd.

Dat de referentie deugt, is apart getoetst voordat er iets gerepareerd werd:
in de menselijk geannoteerde intervallen ligt de EMG-RMS 6,3× (SN1) en 10,2×
(SN2) boven het niveau daarbuiten, tegen 1,44 en 1,03 voor dezelfde events
60 s verschoven. De annotaties liggen exact op het signaal.

## De reparatie

`stap = win / sf` in plaats van de aangenomen 0,1, voor zowel `onset_s` als
`duration_s`. Profielvlag `plm_time_base`.

**Default AAN sinds 21-08-2026** (beslissing van de gebruiker, op advies).
Uitgezonderd blijven `mesa_shhs` en `chicago_1999`, die paper v31/v37
reproduceren; die twee draaien de oude tijdbasis en er ligt een test op die
omvalt als dat verandert.

De vlag volgt daarmee het patroon van `rip_quality_scale_free` — overal aan,
óók op de historische en regulatoire profielen — en níét dat van
`single_channel_rhythm`, dat een scoringscriterium verandert. Het onderscheid
is de reden: geen enkele regelset, hoe oud of hoe regulatoir ook, schrijft
een tijdsafwijking van 2,3 % voor. `aasm_v2_rec`, `aasm_v1_rec` en
`cms_medicare` bootsen oudere criteria na, geen oudere rekenfouten.

Ook de functie-defaults van `analyze_plm` en `_detect_lm_channel` staan op
`True`, zodat wie de bibliotheek rechtstreeks aanroept de juiste tijd krijgt;
`time_base_fix=False` is wat de twee gepinde profielen doorgeven.

Golden 9/9 blijft groen, en dat is echt en niet gemaskeerd: de digest sluit
PLM expliciet uit (`tests/test_golden_output.py`, "Position/snore/PLM/HR are
intentionally excluded"). De respiratoire cijfers bewegen niet mee omdat PLM
de respiratoire scoring niet voedt.

## De afkapping is niet meer stil

Bij deze wissel is `result["events"] = plm_eligible[:200]` blijven staan --
het is een payloadgrens, geen scoringsregel -- maar ze laat nu een spoor na:
`summary["n_events_truncated"]` zegt hoeveel er wegviel, en er komt een
WARNING in het log. Zonder dat kon niemand zien dat de eventlijst ergens
midden in de nacht ophoudt. Tests in `tests/test_plm_event_cap.py`.
