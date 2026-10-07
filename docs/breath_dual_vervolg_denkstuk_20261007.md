# Na `breath_dual` als standaard: fijnafstelling of een ander algoritme — denkstuk

Datum: 2026-10-07. **Denkstuk: er is niets gemeten en niets gebouwd.** Geschreven na de
verschilmeting `docs/breath_vs_breath_dual_eigen_psg_20261007.md` en een claim-trace door
het hele dossier (profiles.py, CHANGELOG, docs, geheugen), zodat niets hieronder opnieuw
voorstelt wat al gemeten of weerlegd is. Productie: `aasm_v3_breath_dual` is sinds vandaag
de standaard voor alle scoorders (YASAFlaskified 0.38.9, `.env` + `instance/config.json`,
per-gebruiker-overrides leeg; rollback = env op `aasm_v3_breath` of `aasm_v3_rec` + `up -d`).

## 1. Wat we weten, en het gat

- **Verschilmeting, 20 eigen PSG's (07-10, geverifieerd):** ΔAHI mediaan 0, maar +5,9 /
  +8,7 / +37,3 /u op de nachten waar de thermistorpoort de thermistor goedkeurt; daar
  vindt `breath` op de thermistor een fractie van de apneus (R18: 32 tegen 458 op de
  druk) en `breath_dual` voegt de druk-apneus toe. RDI daalt onder `breath_dual` op 11/20,
  ventilatoire last springt op de 5 poort-aan-nachten (vier schakelaars, niet één).
- **Referentiebewijs:** alleen MESA n=150 op psgscoring **0.17.0** (14-08, vijftien
  versies oud): `breath` F1 0,510 / bias −5,18; `breath_dual` 0,504 / **−2,34**; `rec`
  0,438 / −5,30. De duale as kocht dus bias (−2,8 /u dichter bij de mens) voor 0,006 F1.
  De oude bevinding "dual is slechter waar de poort doorlaat (46 %)" is op een andere poort
  en versie gemeten en vervalt als bewijs. Op 0.32.0 (n=149 vers) is geen dual-arm gedraaid.
  PSG-IPA heeft één flowkanaal en kan de duale as niet toetsen.
- **Het gat:** de standaard van vandaag heeft **geen actuele referentiemeting**. Dat is
  geen reden om hem niet te kiezen (de bias-winst op 0.17.0 wijst de goede kant op), wel de
  eerste meting die moet volgen (§3.8).

## 2. Het eigenlijke probleem: twee foutmodi op één sensorpaar

- **Foutmodus A — de thermistor is te ongevoelig voor onze drempel.** Gemeten op 597
  menselijke MESA-apneus: de druk zakt mediaan 89,6 %, de thermistor 80,3 %; 50 % tegen
  13 % haalt de drempel van 0,90 bij gelijke AUC (0,918 / 0,941). R11 (0 van 68) en R18
  (32 van 458) zijn precies dat. De vereniging repareert A.
- **Foutmodus B — de druk ziet niets waar wél geademd wordt.** Mondademhaling of een
  verschoven canule geeft een vlakke neusdruk terwijl de thermistor oronasale flow toont.
  Dat is per definitie géén apneu, en de vereniging telt hem wél. Onze apneudefinitie eist
  geen desaturatie, dus niets houdt zo'n event tegen (`docs/flow_artefacten_analyse.md`
  noemt kruiscontrole tegen de tweede sensor en de effortbanden als "wat ik zou bouwen";
  het is nooit gebouwd).
- **R18 ontleed.** Hypopneeën gaan 296 → 72, dus ongeveer 224 van de 426 toegevoegde
  druk-apneus waren onder `breath` al hypopneeën (AHI-neutrale herklassering). De
  resterende ~200 waren onder `breath` **niet gescoord**: ≥ 90 % drukdaling van ≥ 10 s
  zonder desaturatie ≥ 3 % en zonder arousal. Tweehonderd zulke events in één nacht is óf
  een uitzonderlijk fenotype (apneus zonder enig gevolg), óf foutmodus B. **Dat onderscheid
  is meetbaar zonder referentie** — hoe diep zakt de thermistor tijdens die events, en wat
  doen SpO2 en de effortbanden — en het is nog niet gesteld. Het beslist welke knop hieronder
  de juiste is.
- De poort helpt hier niet: `envelope_agreement` meet trage amplitudemodulatie en
  correleert r = +0,07 met werkelijke bevestiging door beide sensoren. Hij zegt "de
  thermistor ademt", niet "de thermistor ziet dezelfde apneus".

## 3. Fijnafstelling — opties, gerangschikt op bewijswaarde per dag werk

| # | optie | dossierstatus | verwacht effect | kost | meetbaar op |
|---|---|---|---|---|---|
| 3.1 | diagnostiek alleen-druk-apneus | open | beslist A vs B | 1 dag, in situ | eigen PSG's (geen referentie nodig) + MESA |
| 3.2 | voorwaardelijke vereniging | **echt open** | B eruit, A gedekt | 2–3 dagen bouw + 1 nacht rekenen | MESA (twee sensoren + NSRR) |
| 3.8 | referentiemeting `breath` vs `breath_dual`, huidige versie | ontbreekt | baseline voor de standaard | harnas bestaat; poortlog toevoegen | MESA n ≥ 150 |
| 3.5 | `hypopnea_strictness` 0,30 onder dual | gemeten op `breath`, nooit onder dual | +F1, bias kan overschieten | knop bestaat; gepaarde arm | MESA, per AHI-tertiel |
| 3.6 | U-Net-arousal inbouwen | gerepliceerd, niet gebouwd (Barts go) | betere arousal-gegradeerde hypopneeën, RDI | 2–3 dagen bouw | PSG-IPA + MESA + SHHS |
| 3.3 | vier schakelaars ontkoppelen | profieldefinitie | RDI/VB gelijk aan `breath` | 1 dag | identiteitstoets op de 20 PSG's |
| 3.4 | thermistordrempel 0,72 onder dual | weerlegd op F1 (single) | alleen FRI-bron en 'both'-aandeel | knop bestaat | MESA |
| 3.7 | poort × drempel als 2×2 | open uit dossier | voorkomt herhaling van de ijkfout | middel | MESA |

**3.1 Diagnostiek eerst, geen knop.** Per alleen-druk-apneu op R02, R11 en R18 (en op
MESA-opnames met NSRR-label, waar "juist/onjuist" wél bekend is): thermistordaling als
fractie van de eigen baseline, SpO2-nadir binnen 30 s, RIP-amplitude. Vooraf vastleggen
wat A en B onderscheidt: A = thermistordaling 50–89 % (samengedrukte schaal, apneu
gemist); B = thermistordaling < 30 % bij behouden effort (de patiënt ademt door de mond).
Een bimodale verdeling is het mooiste antwoord; een eenzijdige maakt 3.2 of 3.4 overbodig.
Recept: zelfde container en `run.py` als de meting van vandaag, met eventlijsten bewaard.

**3.2 Voorwaardelijke vereniging (de kandidaat-fijnafstelling).** Een alleen-druk-apneu
telt mee als (i) de thermistor minstens τ_th zakt — τ_th = 0,72, de sensorafhankelijke
drempel uit `docs/apneudrempel_sensorafhankelijk_bevinding.md`, hier hergebruikt als
**bevestigingsdrempel** in plaats van als eigen apneudrempel (daar was hij F1-neutraal en
bias-halverend) — óf (ii) er volgt een desaturatie ≥ 3 % of een arousal, óf (iii) de
effortbanden tonen het patroon van een obstructie. Alleen-thermistor-apneus bij een
afgekeurde thermistor: alleen met (ii). Foutmodus B verdwijnt, A blijft gedekt. Vooraf te
kiezen, en te verdedigen: **F1 primair, bias mag niet verslechteren** — want de duale as is
al bias-beter, en de les van het 0,72-dossier is dat je de maat niet achteraf wisselt.
Stratificeer op de poort-aan-subset; daar zit het hele verschil. Grens: de NSRR-referentie
is één scoorder met eigen sensorkeuze; `validate_mesa` legt de variant vast (`aasm15`).

**3.8 De ontbrekende referentiemeting.** `breath` tegen `breath_dual` op MESA n ≥ 150 met de
huidige versie, mét per opname de poortuitkomst weggeschreven (dat ontbreekt nu in
`validate_mesa`), per AHI-tertiel. Dit is de baseline waar 3.2 en 3.5 als extra armen in
dezelfde run bij kunnen: **één preregistratie, vier armen** (`breath`, `breath_dual`,
`breath_dual` + voorwaardelijk, `breath_dual` + strictness 0,30), één nacht rekenen op
Obelix (≈ 600 nacht-armen à 3 min op 8 workers).

**3.5 Strictness 0,30 onder dual.** Alle drie de criteria gehaald op `breath` (validatie
n=30 ΔF1 +0,035 op 26/30, bias −3,28 → −0,05), maar onder dual verschuiven hypopneeën
naar apneus (R18 296 → 72), dus de ijking geldt niet automatisch, en met een bias van
−2,34 kan 0,30 doorschieten naar positief. Goedkoop als arm in 3.8; niet los.

**3.6 Het U-Net-arousalmodel is de grootste gevalideerde hefboom die er ligt.** Extern
gerepliceerd (SHHS1 150 verse nachten F1 0,543 → 0,676, p = 2e-22; MESA-76; PSG-IPA 5/5;
CPU 5 s/nacht). Onder `breath`/`breath_dual` worden hypopneeën gegradeerd op arousals in
[t0, t1 + 15 s], en de PSG-IPA-daling van `breath` zit in de arousalclassifier van 0.27.0
(decompositie 07-10): betere arousals raken precies die plek. De doorwerking op AHI,
RDI en `coupled_arousal` is **nooit gemeten** (de replicatie voedde de externe ingang, die
RERA/RDI overslaat). Inbouw als opt-in `arousal_detector="unet_v1"` staat klaar in de
preregistratie van 27-09 en wacht op Barts beslissing (inbouw, default, τ). Daarna is de
meting op `breath_dual` een gewone profielvergelijking.

**3.3 Vier schakelaars ontkoppelen.** `_with_dual_apneas` zet ook `flow_reference =
"hypopnea"` en `thermistor_gate = "respiratory_band"`, met in de docstring een goede reden
(de arousal-analyse leest het referentiekanaal, en onder een additieve thermistor wijst het
apneukanaal naar een niet-getoetste sensor). De schonere vorm: primaire pas op de
poort-gekozen sensor zoals `breath`, tweede pas op de andere sensor, vereniging — dan hoeft
het referentiekanaal niet te wijken en blijven RDI en ventilatoire last gelijk aan `breath`.
Als **nieuw** exploratory profiel (`breath_dual_v2`), de oude definitie bevroren voor de
gepubliceerde cijfers. Identiteitstoets op de 20 PSG's: AHI gelijk aan `breath_dual` op
20/20, RDI gelijk aan `breath` op 20/20. Pas ná 3.8, anders verschuift de baseline onder
de meting. Hoort in dezelfde release als familie `exploratory` → `clinical`: de dropdown
toont de standaard van vandaag met een ⚠ in de groep "experimenteel" en de naam zegt
"(experimental)".

**3.4 en 3.7** alleen als 3.1 foutmodus A als dominant aanwijst en 3.2 niet volstaat; de
drempel 0,72 niet opnieuw als F1-winst voorstellen (weerlegd, −0,0010, p = 0,009), en poort
en drempel alleen samen meten.

## 4. Een ander algoritme

**4.1 Geleerde respiratoire eventdetector.** Er ís een precedent (01-09,
`docs/nacht_20260901_bevindingen.md`): een 1D-U-Net van 462k parameters op 113
trainingsnachten haalde op PSG-IPA mediaan F1 0,539 met dezelfde steile ziektelasthelling
als mensen en regels (0,254 bij < 20 events, 0,743 bij ≥ 150). Dat was een verkenning, geen
poging op schaal. Het arousal-U-Net slaagde pas met 317 verse nachten, een bevroren model
en replicatie op een nooit gezien cohort. Wat een serieuze poging anders moet doen:

- **Data:** MESA 2056 en SHHS1 5792 nachten staan in situ onder DUA; trainen op honderden
  tot duizenden nachten in plaats van 113, met een gescheiden verse set per cohort en
  SHHS1 150 als beslissende externe replicatie (zelfde recept als 27-09).
- **Invoer:** neusdruk, thermistor, SpO2 (met de circulatievertraging als leerbaar
  venster), RIP thorax/abdomen, en de arousalkans uit het EEG-U-Net als extra kanaal; uitgang
  per type (obstructief/centraal/gemengd/hypopneu) zodat de rapportvelden gevuld blijven.
- **Hybride, niet vervangend:** het netwerk als kandidaatgenerator, de AASM-regels
  (duur ≥ 10 s, gradering, desaturatiekoppeling) als nabewerking en als uitleg per event —
  zoals de arousalclassifier nu binnen de regelketen zit. Het rapport moet per event een
  reden blijven dragen; een kaal netwerk-event kan dat niet.
- **Referentie en plafond:** vooraf vastleggen welke NSRR-variant de referentie is
  (`aasm15`, zoals `validate_mesa`), en **eerst het menselijk plafond voor respiratoire
  events op PSG-IPA meten** (12 scoorders, 66 paren per opname; voor arousals is dat 0,679,
  voor respiratoir bestaat het getal nog niet in dit dossier). Zonder plafond is elke F1
  onleesbaar; met plafond weet je of 0,54 → 0,70 nog iets betekent.
- **Risico's die vooraf een antwoord vragen:** (a) de NSRR-DUA en het verspreiden van
  gewichten die uit MESA/SHHS zijn geleerd via PyPI — dezelfde vraag geldt al voor het
  arousal-U-Net-plan en moet uit de DUA-tekst zelf komen; (b) CPU-inferentie op Hetzner
  (het arousalmodel haalt 5 s/nacht, dus haalbaar); (c) de bevroren profielen
  (`mesa_shhs`, `chicago_1999`) blijven regelgebaseerd voor de paperreproductie.
- **Verwachting, eerlijk:** de literatuur rapporteert voor dit soort modellen op
  NSRR-cohorten event-F1's ruim boven onze 0,50–0,54, maar steeds tegen één scoorder en
  met dezelfde zwakte bij lage ziektelast. De winst zit waarschijnlijk in de middentertiel,
  niet in de lichte patiënt.

**4.2 Wat geen algoritme oplost.** Op SN4 scoorde de ene expert één event en de andere
achtendertig (mens-mens F1 0,553). Daar hoort onzekerheid in het rapport, geen betere
detector; dat staat los van de keuze hierboven.

**4.3 De EC-studie is de arbiter voor onze montage.** Alles op MESA is overdracht
(thermistortypes verschillen al binnen MESA: `Therm` 28 / `Aux_AC` 1 op 30). De vraag
"thermistor of druk, en hoe te verenigen" op Somnomedics-montages wordt pas beslist met
menselijk gescoorde eigen nachten; AZORG-YASA-2026-001 (n ≥ 50, gestratificeerd) levert die.
Elke knop hierboven moet daar opnieuw langs.

## 5. Voorgestelde volgorde

1. **3.1** diagnostiek alleen-druk-apneus — één dag, geen referentie nodig, beslist A tegen B
   en dus welke knop zinvol is. Preregistratie van de verdelingscriteria vooraf.
2. **3.8 + 3.2 + 3.5 in één MESA-preregistratie, vier armen** — de ontbrekende baseline
   voor de standaard van vandaag plus de twee goedkoopste knoppen. Eén nacht rekenen.
3. **3.6** U-Net-arousal inbouwen als opt-in zodra Bart beslist; daarna doorwerking op
   `breath_dual` meten (AHI, RDI, `coupled_arousal`).
4. **4.1** scout onder `bench/resp/` parallel, met als voorwerk het respiratoire menselijke
   plafond op PSG-IPA en de DUA-check.
5. **3.3 + familie clinical** in één release, ná stap 2.

Wat ik hiervoor níet heb gedaan: geen meting, geen code, geen wijziging aan profielen.
De enige wijziging van vandaag is de productiestandaard.
