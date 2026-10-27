# Plan — humaan tegen machine vergelijken per opname en over de studie (EDFbrowser als viewer)

Datum: 2026-10-27. Status: voorstel voor Bart; niets gebouwd. Doel: één werkwijze waarmee een
menselijke scoring en de psgscoring-scoring van dezelfde opname (a) event voor event bekeken kunnen
worden in EDFbrowser en (b) met vooraf vastgelegde maten worden samengevat, per opname en over de
EC-studie AZORG-YASA-2026-001 (n ≥ 50, gestratificeerd).

## 0. Wat eerst vast moet liggen (blokkerend, bij Bart)

1. **Het menselijke invoerformaat**: één geanonimiseerde Somnomedics-eventexport (of de
   formaatspecificatie): kolommen, tijdbasis (klok of seconden vanaf start), eventlabels, en of
   slaapstadia en arousals meekomen. Zonder voorbeeld kan de parser niet getest worden.
2. **De menselijke hypopneuregel**: 3 % of 4 % desaturatie, met of zonder arousal. Anders meet de
   vergelijking een regelverschil en noemt het een machinefout. De machine-arm draait het profiel
   dat die regel implementeert (`aasm_v3_breath_dual`, regel 1A).
3. **De primaire koppelregel**: IoU ≥ 0,2 (projectstandaard) primair, "elke overlap" secundair.
   Een F1 zonder koppelregel is geen F1.
4. **De machineconfiguratie van de studie-arm**: psgscoring-versie gepind en de beslissing over
   `stage_baseline` (26/27-10) vóór de eerste studienacht genomen en in het protocol-addendum
   vastgelegd; sinds 12-10 scoort productie stilzwijgend anders dan de maanden ervoor.

## 1. psgscoring: `agreement` uitbreiden (bibliotheek, 1 dag)

Bestaat al: `compare_event_sets` (IoU/elke overlap, TP/FP/FN, Jaccard, `kappa_1s`, onset-/eind-
verschuiving, mediane duur). Erbij, elk met tests die eerst falen:

- `ppa`/`ppv` als benoemde velden (TP/H en TP/M) naast F1; per eventtype (apneu O/C/M, hypopneu,
  arousal) en totaal.
- Indexniveau: `bland_altman(h, m)` → bias, SD, 95 %-limieten; `icc_2_1(h, m)`; gewogen κ en
  4×4-matrix voor de ernstklassen (normaal/licht/matig/ernstig) met "exact" en "binnen één klasse".
- `agreement_card(events_h, events_m, hypno, ahi_h, ahi_m, ...)` → één dict per opname: H/M/TP/
  FP/FN, PPA, PPV, F1, κ_1s, verschuivingen, AHI H → M met Δ, ernst H → M, per type; plus de
  lijst FP- en FN-events (voor de viewer).
- Plafondtabel als data: menselijke F1 op PSG-IPA (respiratoir 0,667, arousal 0,679; 330 paren),
  zodat elke kaart "x % van het menselijk plafond" kan tonen.
- Release als 0.38.0 (alleen nieuwe functies; geen scoringsgedrag), na de gebruikelijke poort.

## 2. YASAFlaskified: invoer, kaart en export (1 à 2 dagen)

- **Invoer**: upload van het menselijke eventbestand bij een bestaande job (parser voor het
  Somnomedics-formaat uit stap 0; tijdbasis naar seconden vanaf opnamestart; validatie: events
  binnen de opname, labels bekend). Opslag naast de job op de server; geen patiëntgegevens in
  het bestand nodig (events, geen namen).
- **Kaart**: resultatenpagina krijgt een sectie "Overeenstemming met menselijke scoring" (alleen
  als er een menselijk bestand is): de kaart uit stap 1, per type, met plafond en koppelregel in
  het bijschrift. Niet in het klinische PDF (studie-instrument, geen verslagonderdeel), tenzij Bart
  het wél wil.
- **Export voor EDFbrowser** (downloadknop, één zip): (i) machine-events, (ii) menselijke events,
  (iii) alleen-FP en alleen-FN, (iv) YASA-hypnogram als 30 s-stadiumannotaties — als CSV
  (onset s, duur s, label) met voorvoegsels `M:`, `H:`, `FP:`, `FN:` in het label, en als
  EDFbrowser-XML. Leveringstests volgens de huisregel (het veld is pas geleverd als de pagina en
  de export het lezen).
- Tests eerst; CI-poort (ruff/mypy) zoals sinds 26-10.

## 3. EDFbrowser-werkwijze (halve dag)

- Eén pagina `docs/edfbrowser_vergelijking.md`: EDF openen, CSV importeren (kolomtoewijzing,
  tijdbasis "seconden vanaf start", één keer instellen), annotatielijst gebruiken om door FP/FN te
  springen, hypnogram tonen uit de stadiumannotaties, filters voor flow/EEG.
- Proef op de demo-EDF met een synthetische "menselijke" lijst (verschoven en ontbrekende events)
  en op één eigen nacht (met toestemming): klopt de tijdbasis op de seconde, springt de lijst
  naar het juiste event, toont het hypnogram.
- Geen fork, geen GPL-code in de BSD-repo's; alleen bestandsuitwisseling.

## 4. Studieniveau: analyse-script en preregistratie (1 dag + prereg vóór de eerste nacht)

- `scripts/agreement_study.py`: over alle jobs met menselijke scoring → tabellen per type en per
  tertiel van de menselijke AHI, Bland-Altman voor AHI/RDI/arousal-index, ernstmatrix met gewogen
  κ, F1-boxplot met de plafondlijn, ICC; uitvoer als markdown + json (geen patiëntgegevens).
- **Preregistratie van de studie-analyse** vóór de eerste gescoorde nacht wordt bekeken: de maten
  van het EC-protocol (AHI-bias, r, κ, F1) als primair, PPA/PPV/ICC/limieten als secundair;
  koppelregel; hypopneuregel; tertielen; wat een "klinisch relevant verschil" is (voorstel:
  ernstklasse-wissel of |ΔAHI| ≥ 5 /u); hoe met één scoorder per nacht wordt omgegaan (geen
  consensus, dus de scoorder is de referentie en het plafond komt uit PSG-IPA).
- Elke opname registreren (welke nachten, welke versie, welke scoorder-id) in een register op de
  server, mode 600.

## 5. Validatie van het instrument zelf (halve dag)

- PSG-IPA (5 opnames, 12 scoorders) als proefbank: de kaart moet de bekende getallen reproduceren
  (protocolwaarde F1 0,890 op de referentieversie; plafond 0,667) en het bestaande harnas
  (`validate_psgipa`) byte-gelijk nazeggen — één bron van waarheid voor scoordersdata, het harnas
  blijft het studieartefact.
- Meting-verificatie-agent over de eerste studie-uitvoer.

## 6. Randvoorwaarden

- Geen patiëntgegevens in exports, logs of git; koppeling job ↔ opname blijft op de server.
- Machineconfiguratie bevroren per studiefase; elke wijziging van psgscoring tijdens de studie =
  protocol-addendum, niet stil.
- Licenties: EDFbrowser GPL v3 blijft een los programma; psgscoring/YF blijven BSD-3.

## Volgorde en inzet

0 (Bart) → 1 → 2 → 3 → 5 → 4. Totaal ongeveer vier werkdagen plus de prereg; stap 1 en 3 kunnen
morgen starten zodra het voorbeeldbestand en de hypopneuregel er zijn. Beslissingen bij Bart:
punt 0.1–0.4, kaart wel/niet in het PDF, en of de studie-arm `stage_baseline` aan of uit draait.
