# Preregistratie — `breath`, `breath_dual` en de voorwaardelijke vereniging op MESA (stap 2)

Datum: 2026-10-07, **geschreven vóór de run.** Vervolg op het denkstuk
(`docs/breath_dual_vervolg_denkstuk_20261007.md` §3.2/3.5/3.8) en op stap 1
(`docs/alleen_druk_apneus_diagnostiek_preregistratie_20261007.md`; de uitkomst daarvan
staat in `docs/alleen_druk_apneus_diagnostiek_20261007.md` en verandert de regels hieronder
niet — alleen de duiding). `aasm_v3_breath_dual` is sinds 07-10 de productiestandaard
zonder actuele referentiemeting; dit is die meting, plus de twee goedkoopste knoppen.

## Cohort
- **MESA is op (2055 van 2055 nachten ooit gebruikt; SHHS heeft één flowkanaal).** Daarom
  de **standaard-n150-validatieset** (seed 20260801 uit `validate_mesa`, dezelfde trekking
  als 14-08 waar de enige dual-referentie op 0.17.0 vandaan komt; het register zegt dat
  daar niets op is afgesteld), **minus de 10 nachten** die ook in een kalibratietrekking
  zitten (0,72: seed 20260823 n=30 na uitsluiting van de eerste 50; strictness: seeds
  20260824 n=30, 20260825 n=15, 20260826 n=15, in alle reconstructies van de uitsluiting)
  → **n = 140**, lijst `docs/breath_dual_mesa_20261007/opnames.txt`, geregistreerd in
  `/srv/DATA/MESA/gebruikte_mesa_ids.txt`.
- **Hergebruik, benoemd:** de eerste 50 van deze set waren de validatieset van de
  0,72-drempel (22-08, weerlegd op F1) en de hele set de 0.17.0-vergelijking. Geen
  parameter van de armen hieronder is op deze nachten gekozen. Gevoeligheidsset voor de
  primaire vergelijking: de **93** nachten uit posities 51–150 zonder kalibratie-overlap.
- Referentie: NSRR `aasm15` zoals `validate_mesa` (één scoorder), matcher
  `LEGACY_MATCHER` (IoU 0,20, typeonbewust); artefact-epochs leeg (default van het harnas,
  zoals bij de 0.17.0-vergelijking); hypnogram uit de nsrr-xml.

## Armen (één run, `scripts/validate_mesa.py --profiles aasm_v3_breath aasm_v3_breath_dual --strictness 0.5 0.3 --dual-confirmation`)
`rec` (anker, altijd mee), `breath@0,50`, `breath@0,30`, `dual@0,50`, `dual+conf@0,50`,
`dual@0,30`, `dual+conf@0,30`. `+conf` = `PSGSCORING_DUAL_SENSOR_CONFIRMATION=
thermistor_or_consequence` met de defaults (thermistordaling ≥ 0,72, desaturatie ≥ 3 %,
arousal in [t0, t1 + 15 s]); de bibliotheek is bevroren vóór de start (commit in het
verslag) en wordt tijdens de run niet aangeraakt.

## Beslisregels (vooraf)
1. **Primair — `dual+conf@0,50` tegen `dual@0,50`, F1 primair:** gepaarde ΔF1 mediaan
   > 0 **én** beter op meer nachten dan slechter **én** Wilcoxon p < 0,05 (nullen
   weggelaten), **én** bewaker: de gemiddelde AHI-bias van `+conf` ligt niet meer dan
   1,0 /u verder van nul dan die van `dual`. Haalt hij dat: de voorwaardelijke
   vereniging is een kandidaat voor `breath_dual` (aanzetten = Barts beslissing met
   klinische aan/uit-controle). Haalt hij het niet: gebouwd-uit, cijfers in CHANGELOG.
2. **Baseline — `dual@0,50` tegen `breath@0,50`:** geen beslisregel (productie is al
   omgezet); rapporteren: ΔF1, bias, MAE, ernstklasse-overeenstemming, per
   NSRR-AHI-tertiel, en of het teken van 0.17.0 (bias beter, F1 −0,006) standhoudt.
3. **Secundair — strictness 0,30 onder dual (`dual@0,30` tegen `dual@0,50`):** de regel
   van 24-08: mediane gepaarde ΔF1 ≥ +0,010, p < 0,05, |gemiddelde bias| niet meer dan
   1,0 /u slechter. Idem voor `dual+conf@0,30` tegen `dual+conf@0,50`. Ook hier volgt
   niets automatisch.
4. **Per event (beschrijvend, de kern van de vraag):** elke alleen-druk-apneu van
   `dual@0,50` wordt tegen de NSRR-apneus gelegd (IoU ≥ 0,20): aandeel dat een menselijke
   apneu matcht, per klasse A/C/B (thermistordaling ≥ 0,72 / 0,30–0,72 / < 0,30) en per
   bevestiging (thermistor / desaturatie / arousal / vervallen). Verwachting die de
   regel rechtvaardigt: vervallen events matchen zelden een NSRR-apneu, bevestigde vaak.
   Rapporteren, geen criterium.

## Rekenplan
Z6, 20 workers (≈ 6 GB elk), 7 armen × 140 nachten à ~3 min ≈ 2,5–3,5 h; checkpoint
per nacht (`.partial.jsonl`), thermal guard gebonden aan de pgid met geverifieerde
logregel. Uitvoer `docs/breath_dual_mesa_20261007/mesa.json` + analyse-script, verslag
`docs/breath_dual_mesa_20261007.md`, daarna meting-verificatie.
