# Preregistratie — klinische aan/uit-controle van de U-Net-arousaldetector op de 20 eigen PSG's

Datum: 2026-10-26, geschreven vóór de meting. Opdracht van Bart na de uitrol van YASAFlaskified
0.39.1 / psgscoring 0.37.0 (`unet_v1` standaard op de klinische profielen): "doe de klinische
controle, je hebt permissie" (dekt het lezen van de eigen opnames op de server, runbook §7).

## Vraag

Wat verandert de nieuwe standaard `unet_v1` — en de opt-in `unet_v2` — aan de productie-uitkomsten
op dezelfde 20 eigen diagnostische PSG's ten opzichte van de LGBM-keten van vóór 25-10, en, eerst:
komt het U-Net in de productieroute überhaupt aan het werk?

## Leveringsvraag vooraf (uit de code gelezen, vóór de meting)

`myproject/tasks.py::_pneumo_load_plan` laadt voor de pneumo-raw de respiratoire kanalen, het
stagings-EEG, de kin-EMG en de arousal-afleidingen, maar **niet het EOG**; `_pneumo_channel_map`
geeft geen sleutel `"eog"` door. psgscoring 0.37.0 zoekt het EOG voor het U-Net in `raw.ch_names`
(`pipeline._pick_eog`: geconfigureerde naam, anders patroon EOG/E1-/E2-/LOC/ROC) en valt zonder EOG
terug op de LGBM-keten met `summary["detector"] = "lgbm"` en
`summary["unet_fallback_reason"] = "geen EOG-kanaal (…)"` (`arousal.py`, v0.35.0-blok). De
rapportlaag toont die terugval in de Herkomst-rij ("LGBM-keten (lgbm) — U-Net niet gebruikt: …").
**Verwachting: arm P (productieroute ongewijzigd) valt op 20/20 terug en is op elk veld gelijk aan
arm A.** Dat wordt gemeten, niet aangenomen; draait P wél het U-Net, dan is deze lezing fout en
wordt P ≡ B verwacht. Er zijn sinds de uitrol van 10:40 geen productiejobs gedraaid, dus er is nog
geen verslag dat dit al laat zien.

## Steekproef en invoer

Dezelfde 20 opnames als op 07-10 en 10-10 (R01–R20; `mapping.json` ongewijzigd gekopieerd,
koppeling job-id ↔ R blijft op de server in `/data/slaapkliniek/metingen/unet_klinisch_20261026/`,
mode 600; alle 20 jobconfigs, resultaten en EDF's zijn aanwezig, alle 20 hebben een `eog_ch` in de
jobconfig). Per opname dezelfde invoer als de worker: opgeslagen YASA-hypnogram, opgeslagen
artefact-epochs, dezelfde kanaalkeuze en inleesroute (`_detect_pneumo_channels`,
`_pneumo_load_plan`, `_load_edf`, `_pneumo_channel_map`, `_position_code_map` uit `tasks.py` van
het image). Losse container uit het productie-image `yasaflaskified:0.39.1` (psgscoring
**0.37.0**, versiecontrole in `run.py`), uploads read-only, 16 van 32 CPU's, `nice -n 10`,
8 analyses parallel, `OMP_NUM_THREADS=2`; alleen `YASAFLASKIFIED_POSITION_CODE_MAP` uit de
productie-`.env` gaat mee (geen geheimen). Jobqueue leeg bij de start; app- en workercontainers
blijven onaangeraakt; bereikbaarheid van de app elke 60 s gelogd. Profiel op alle armen
`aasm_v3_breath_dual` (productiestandaard), `split_night="off"`. Werkpunt niet gezet: het eigen
werkpunt van het gewicht (v1 0,35, v2 0,30; `PSGSCORING_AROUSAL_UNET_THRESHOLD` afwezig).

## Armen (per opname, zelfde hypnogram en artefacten)

| arm | inleesroute | detector | rol |
|---|---|---|---|
| P | productie (laadplan zónder EOG) | profieldefault (`unet_v1`), geen env | **levering**: wat de app sinds 10:40 doet |
| A | productie + EOG (`eog_ch` uit de jobconfig, `"eog"` in de channel_map) | env `lgbm` | keten van vóór 25-10 — referentiearm |
| B | idem | env `unet_v1` | nieuwe standaard — **primair** |
| C | idem | env `unet_v2` | opt-in kandidaat — secundair |

`PSGSCORING_AROUSAL_DETECTOR` wordt per aanroep gelezen (`pipeline._arousal_detector`), wordt
vóór elke arm gezet en erna verwijderd (voor P afwezig); de gelezen waarde staat per arm in de
uitvoer als `summary.detector`, met `unet_threshold`, `unet_onnx_sha256`, `unet_fallback_reason`
en `unet_t_total_s`. Arm P laadt zijn raw apart met het productielaadplan (geen kanaal eruit
droppen: een andere kanaalset kan een andere inleesroute geven); A, B en C delen één raw met het
EOG erbij.

## Vooraf vastgelegde lezing

1. **Levering (P) eerst.** Per nacht `detector` en `unet_fallback_reason`. Terugval op 20/20 →
   de nieuwe standaard komt in productie niet aan; dan moet P op elk veld gelijk zijn aan A
   (identiteit 20/20; anders is er méér aan de hand dan het ontbrekende EOG en wordt dat eerst
   uitgezocht), en de reparatie (EOG in laadplan én channel_map, test die eerst faalt) gaat als
   YASAFlaskified 0.39.2 — pas na Barts akkoord, niet uit deze meting. Terugval op 0/20 → P moet
   op elk veld gelijk zijn aan B. Iets ertussen: per nacht verklaren.
2. **Terugval in A/B/C.** Verwacht geen (alle 20 hebben een `eog_ch`; of die naam in het EDF
   zit wordt per nacht vastgelegd). Elke terugval wordt met reden gemeld; die nacht blijft in de
   tabellen staan maar valt uit de B/C-statistiek.
3. **Primair: B tegen A**, per opname: arousaltelling en arousal-index, respiratoire
   arousal-index, RERA-telling en -index, RDI, AHI, hypopneeën, apneus O/C/M, ernstklasse
   (normaal/licht/matig/ernstig), ventilatoire last; plus de overeenstemming van de arousallijsten
   B tegen A (`agreement.compare_event_sets`, `matcher="any"`: F1, precisie/recall van B tegen A
   als referentie, κ_1s — beschrijvend, A is geen waarheid). Per opname én per tertiel van de
   opgeslagen AHI (grenzen van 07-10).
4. **Secundair: C tegen B**, zelfde opmaak; C tegen A alleen samengevat.
5. **Verwachtingen** (geen slaagcriteria; verschilmeting zonder menselijke referentie):
   - apneus O/C/M identiek op 20/20 over A, B en C (de arousallijst bereikt op `breath_dual`
     alleen de hypopneeën via de arousalpoot van de 3 %-regel, de bevestiging van enkelsensor-
     apneus en de RERA/FRI-lijst; een apneu-verschil is onverwacht en wordt uitgezocht);
   - AHI: mediaan |Δ| < 1 /u (PSG-IPA 09-10: ±0,6 /u; MESA n=100: −0,2 /u mediaan), nachten met
     |ΔAHI| ≥ 5 /u: 0 à 1;
   - arousal-index onder `unet_v1` lager op de meerderheid (MESA: 73/100 lager, mediaan −1,8 /u,
     laag tertiel −29 arousals; PSG-IPA SN2 137 → 87), RDI volgt de arousaltelling;
   - `unet_v2` tegen `unet_v1`: kleine verschuiving (PSG-IPA ΔF1 +0,008 mediaan), telling niet
     systematisch hoger of lager (τ 0,30 is het tellingsneutrale werkpunt van v2);
   - rekentijd U-Net per nacht enkele seconden op 2 threads (`unet_t_total_s`).
6. **Prominent in het verslag, voor Barts beoordeling:** elke nacht met |ΔAHI| ≥ 5 /u, elke
   ernstklasse-wissel, elke nacht waar de arousal-index meer dan 50 % verandert, elke terugval
   of fout, en elke afwijking van de verwachtingen hierboven.
7. **Geen default-wissel uit deze meting.** Er is geen referentie; het hoogst haalbare is "niet
   slechter én verklaarbaar". Uitrol van een EOG-reparatie, en of `unet_v2` ergens standaard
   wordt, zijn Barts beslissingen.
8. Statistiek: mediaan, gemiddelde, bereik; exacte tekentoets op de niet-nulverschillen (bij
   weinig niet-nulverschillen zegt die alleen iets over het teken); per tertiel.

## Uitvoer en verificatie

Per opname `R*.json` zonder patiëntgegevens: tellingen, samenvattingen, provenance en de
eventlijsten (arousals en respiratoire events als onset/duur/type in seconden vanaf opnamestart;
geen kloktijden, namen, paden of job-id's). Kopie naar `/srv/CODE/docs/unet_klinisch_20261026/`;
analyse met een lokaal `analyse.py`; daarna de meting-verificatie-agent. Verslag:
`docs/unet_klinische_controle_eigen_psg_20261026.md`.
