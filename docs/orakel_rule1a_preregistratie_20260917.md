# Preregistratie — orakel-decompositie van de Rule-1A-arousaltak

Datum: 2026-09-17. **Geschreven vóór enige meting.** Keten: psgscoring
0.34.2 (git `8dac397`), `aasm_v3_rec`, arousalketen zoals uitgerold
(multi-derivatie-union, LGBM-werkpunt 0,70 + 10 s-interval, autonome
re-ranker waar Pleth/HR aanwezig).

## Vraag

Op 29-08 is de arousaltak van Rule 1A weerlegd (MESA n=150 vs `aasm15`:
F1 0,438→0,382, precisie 0,517→0,342, bias −5,26→+8,01). De diagnose
destijds: "de tak vermenigvuldigt de fout in zijn invoer". Dat laat twee
oorzaken open die verschillend werk vragen:

1. **onze arousals** — te veel fout-positieven op precies de plekken
   waar afgewezen kandidaten liggen (arousal-F1 0,55 tegen plafond 0,68);
2. **de koppelregel zelf** — venster 15 s (gemeten-goed, 21-08), gap
   ≤ 1 ademteug, eligibility (`no_desaturation`) — te toegeeflijk, ook
   met perfecte arousals.

De decompositie vervangt in één arm onze arousals door de
referentie-arousals en houdt al het andere gelijk.

## Armen (alle in-pipeline, zelfde raw/hypno/profiel)

| arm | tak | arousalbron | mechanisme |
|---|---|---|---|
| A | uit | — | productie (`PSGSCORING_RULE1A_AROUSAL=0`) |
| B | aan | onze detector | `PSGSCORING_RULE1A_AROUSAL=1` + `PSGSCORING_AROUSAL_LIMB_WIRED=1` |
| C | aan | **referentie** | `run_pneumo_analysis(arousal_events=ref)` — bron "external" wordt altijd gehonoreerd — + `PSGSCORING_RULE1A_AROUSAL=1` |

Referentie-arousals: PSG-IPA — de vaste EEG-arousalannotatie per opname
(set van scoorderbestand 1; SN2–4 verschillen per set één event); MESA —
NSRR `Arousals|Arousals` uit de nsrr-xml (start + duur).

## Cohorten

* Fase 0: PSG-IPA n=5, referentie = 12 scoorders (validate_psgipa-conventies:
  hypnogram scoorder 1, matcher IoU 0,20, F1 = mediaan over scoorders).
  Descriptief; tevens getrouwheidscontrole: arm A moet de paper-v31-AHI's
  bit-identiek reproduceren (8,1 / 9,3 / 53,8 / 4,3 / 11,0).
* MESA: de standaard held-out n=150 (seed 20260801, dezelfde 150 als 29-08),
  referenties `aasm15` (primair) en `desat3_all` (secundair), matcher
  `LEGACY_MATCHER`, geen artefact-epochs (zoals 29-08). Niets wordt hierop
  afgesteld; de set wordt in `gebruikte_mesa_ids.txt` geregistreerd.

## Uitlezingen (per opname, per arm)

* AHI, n events, F1/precisie/recall vs referentie.
* **R1 herstelprecisie**: events die in B resp. C wél en in A níet staan zijn
  de herstellingen; hun precisie = aandeel dat een referentie-event matcht.
* **R2 gepaarde ΔF1** (C−A, B−A) vs `aasm15`: mediaan, aantal beter,
  Wilcoxon (`validate_mesa.wilcoxon_signed_rank`).
* **R3 structurele-gat-recall**: referentie-events in `aasm15` die NIET in
  `desat3_all` zitten zijn de arousal-only-hypopneus (het gat dat de tak
  moet dichten); aandeel daarvan dat C resp. B matcht.
* Bias en ernstklasse-verschuivingen als bewaker.

## Beslisregel (vooraf)

Op MESA n=150, primair `aasm15`:

* **Koppeling deugt, detector is de flessenhals** — als R1(C) ≥ 0,60 ÉN
  ΔF1(C−A) > 0 op ≥ 90/150 met Wilcoxon p < 0,05. Gevolg: de tak blijft
  uit; vervolgwerk richt zich op arousalprecisie op koppelplekken, niet op
  de koppelregel.
* **Koppeling te toegeeflijk** — als R1(C) < 0,50 ÓF ΔF1(C−A) ≤ 0. Gevolg:
  de koppelregel (gap/eligibility/latentie) moet herontworpen worden vóór
  enig verder takwerk; de arousaldetector is niet (alleen) het probleem.
* Daartussen: rapporteren, geen besluit.

Arm B is een herhaling van het 05-09-dossier op de huidige keten (open
punt aldaar); verwachting uit 29-08: ΔF1(B−A) < 0. Geen besluit over B
volgt uit deze meting — hij dient als ijk voor het verschil met C.

## Rekenplan

150 nachten × 3 armen, 20 workers × 1 BLAS-thread (~6 GB/worker, 151 GB
RAM), temperatuurbewaker op `x86_pkg_temp` (drempel 85 °C, 3× ≥ → stop).
Harnas: `scripts/orakel_rule1a.py`; uitvoer `docs/orakel_rule1a_*.json`.
