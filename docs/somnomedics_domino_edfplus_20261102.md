# Hoe SOMNOmedics (DOMINO) EDF+ maakt — en wat dat betekent voor onze eigen EDF+ (02-11-2026)

Vraag van Bart (02-11): "zoek op hoe SomnoMedics EDF+ genereert", naast het genereren van EDF+ voor de
herrekende studies. Bronnen: de DOMINO-handleiding 2.6.0 (hoofdstuk 13 *Data exchange*), de
SOMNOmedics-kennisbank, de Snooz-documentatie over DOMINO-ASCII-bestanden, en de headers van onze eigen
uploads op de server (alleen structuur, geen identificatievelden). Geen patiëntgegevens in dit document.

## 1. Wat DOMINO exporteert

**EDF-export = ruwe signalen, geen scoring.** Handleiding §13.1.1 kent twee wegen:

- *ongefilterde* ruwe data: in het dialoogvenster "Open Recording" de meting selecteren (of rechtsklikken)
  → "EDF Export" (zo ook in de kennisbank "How to export raw data as EDF-file?");
- *gefilterde of gerefereerde* ruwe data: meting openen → "File" → "Export raw data" → "EDF Export".

Bij de overdracht van de flashkaart kan "EDF Export" als vinkje mee (§7.1.7), dus een export kan al bij
het inlezen ontstaan. In het exportvenster (fig. 13-2) kies je per kanaal of het meegaat, en per kanaal
het EDF+-label: "?" = geen EDF+-label, "v" = label uit de vertaaltabel, handmatig toegekend kan ook.
Per kanaal zijn eenheid, fysiek en digitaal minimum/maximum, prefiltering en transducer instelbaar
(fig. 13-4); de vertaaltabel en de kanaalkeuze zijn als sjabloon te bewaren. **Nergens in §13.1.1 staat
iets over events, slaapstadia of markers**; de handleiding noemt EDF+ alleen om "data te delen met andere
slaaplaboratoria of apparaten".

**Scoring gaat via een andere export.** §13.1.2: "File" → "Export analysis data" geeft analysedata (het
slaapprofiel, flow-events, …) als beeld, RIFF of **ASCII**, per kanaal, met een exportfrequentie en
optioneel de tijdbasis of het slaapprofiel erbij. Dat zijn de "DOMINO ASCII accessory files" die
Snooz en SpiSOP converteren. Vorm (uit de Snooz-documentatie):

```
Signal ID: SchlafProfil
Start Time: 08/03/2016 0:14:00
Unit:
Signal Type: Discret
Events list: N4,N3,N2,N1,REM,Wake,Movement
Rate: 30 s
00:14:00,000; Wake
```

en voor events (`Signal Type: Impuls`, bv. `Signal ID: FlowD`): `00:22:48,246-00:23:00,246; 12;Hypopnea`
(begin–einde, een getal, label). Eén bestand per analysespoor (slaapprofiel, flow-events, arousals, PLM,
snurken, SpO2-events); de cijferreeksen (hartslag, SpO2, positie) hebben `Signal Type: Analog`.

**Lichaamspositie.** Kennisbank "Exporting the body position as EDF": de EDF-waarde is "directly the
hexadecimal value of the sensor" en wordt geklasseerd als ≤ 172h = buik = **1**, > 172h = rechtop = **2**,
> 3c0h = links = **3**, > 6e8h = rechts = **4**, > 980h = rechtop (toestel omgekeerd) = **5**, > a70h =
rug = **6**. Dat is exact de tabel die sinds 18-10 op de server staat
(`YASAFLASKIFIED_POSITION_CODE_MAP`, methode `coded` in de resultaten) — de leverancier bevestigt de
keuze van 18-10.

**Import.** §13.2 "EDF import": bestand kiezen, een hardwareconfiguratie kiezen, per kanaal een
hardwarekanaal, label en bemonsteringsfrequentie toekennen, importeren. Alleen kanalen; over annotaties,
slaapstadia of events zegt de handleiding niets. De kennisbank beschrijft voor "vreemde" EDF's een omweg
via Polyman (kanalen samenvoegen, als .rec/.edf exporteren, "DOMINO can import .rec files as well as
.edf files"). Conclusie: **onze gescoorde EDF+ kan DOMINO wel als signalen inlezen, maar de scoring erin
niet** — althans niet volgens de documentatie tot en met 3.0; voor nieuwere versies is dat een vraag aan
SOMNOmedics.

## 2. Wat onze uploads laten zien (headers, server)

De EDF's die de kliniek uploadt, hebben allemaal dezelfde bouw:

- `EDF+C`, records van **1 s**, versie "0", één kanaal `EDF Annotations` (60 samples = 120 bytes per record);
- **maar twee annotaties in de hele nacht: `Start` (bij +0,75 s) en `Lights Off`.** Geen slaapstadia, geen
  events — consistent met §1: DOMINO's EDF+ draagt alleen markers;
- gerefereerde afleidingen (`EOG1:A2`, `F4:A1`, `C4:A1`, `O2:A1`, `F3:A2`, `C3:A2`, `O1:A2`, zonder
  prefilterveld) náást de ruwe elektroden (`C3`, `C4`, `O1`, `O2`, `A1`, `A2`, `F3`, `F4`, `EOG1`, `EOG2`
  met `HP:0.2Hz LP:35.0Hz`) — de export van gefilterde/gerefereerde data uit §13.1.1;
- `Pressure Flow` (256 Hz, `LP:1.0Hz`), `Flow Th.` (32 Hz), `RIP Thora`/`RIP Abdom`/`Sum RIP` (32 Hz),
  `Snore`/`Pressure Snore` (256 Hz), `Pleth` (128 Hz): **zonder fysieke eenheid** (leeg veld), fysiek
  ±1000 op 16 bit; `SpO2` en `Pulse` 0–255 fysiek én digitaal (8-bitwaarden in een 16-bitveld);
  `Pos.` 0–10 (de codes 1–6 hierboven); `PLMl`/`PLMr` 128 Hz; `EMG1..3` 256 Hz; `ECG II` 256 Hz;
  plus `Light` (lux), `Accu`, `Accu Stan`, `Activity` (mg);
- de identificatievelden (patiënt, opname) zijn gevuld — daarom blijft de brondata op de server.

Het veld `Lights Off` is een bruikbare marker die we nu niet lezen (analysevenster/TIB); apart
voorstel, geen onderdeel van deze notitie.

## 3. Hoe onze eigen EDF+ eruitziet (YASAFlaskified `generate_edfplus.py`)

- Schrijver: pyedflib `FILETYPE_EDFPLUS` (→ `EDF+C`), MNE en edfio als terugval; alle bronsignalen worden
  gekopieerd met label, frequentie, eenheid, bereik, transducer en prefilter. Een kanaal zonder eenheid
  krijgt van pyedflib de standaard `mV` — kosmetisch verschil met de bron. De annotaties van de bron
  (`Start`, `Lights Off`) gaan **niet** mee; de schrijver bouwt de annotatielijst uit de resultaten.
- Annotaties (ASCII, max. 80 tekens): `Sleep stage W|N1|N2|N3|R` per 30 s (de conventie van EDFbrowser en
  Sleep-EDF), `Obstructive apnea|Central apnea|Mixed apnea|Hypopnea|Central hypopnea|Obstructive hypopnea`
  met `desat=…% conf=…`, `SpO2 desaturation …%` (max. 200), `Arousal`, `Arousal (respiratory)`, `RERA`,
  `Arousal (PLM)`, `PLM` (`(resp-assoc)`), `Artifact` per epoch, `Position -> …`, `CSR detected (…)`.
- Bestand: `uploads/<job>_scored.edf`, in de pipeline gemaakt (stap EDF+ in `run_analysis_job`), download
  via `/results/<job_id>/edfplus` (bestandsnaam `<patiëntnaam>_scored.edf`; on-the-fly als het ontbreekt).
- Lezen door derden: EDFbrowser toont de annotaties en bouwt het hypnogram via zijn labelmapping
  (dialoog "hypnogram": labels per stadium koppelen; "Use annotations' duration for epoch length");
  Tools → Import/Export annotations ondersteunt CSV, XML en EDF+. Voor DOMINO geldt §1: signalen ja,
  scoring nee.

## 4. Wat hieruit volgt

1. Een SOMNOmedics-EDF+ is een signaalcontainer met twee markers; wie de menselijke scoring wil
   vergelijken, moet de **ASCII-analyse-export** vragen (slaapprofiel + eventbestanden), niet de EDF.
   Dat is ook de weg naar menselijke events met onsets voor de eigen nachten (nu alleen een AHI).
2. Onze `_scored.edf` is voor een menselijke reviewer bruikbaar in EDFbrowser; terug in DOMINO kan alleen
   het signaal. Vraag aan SOMNOmedics (DOMINO ≥ 3.x): is er een import van EDF+-annotaties of van de
   eigen ASCII-eventbestanden?
3. Twee kleine verbeterpunten voor de schrijver, geen van beide dringend: bron-annotaties (`Lights Off`)
   meenemen, en een lege eenheid leeg laten in plaats van `mV`.

## Bronnen

- DOMINO Instruction Manual 2.6.0 (SOMNOmedics), §7.1.7, §8, §13.1–13.2:
  https://support.somnomedics.de/wp-content/uploads/2020/02/Manual_-DOMINO_2.6.0_eng_Rev_2.pdf
- DOMINO 3.0.0 manual (geen EDF-paragraaf):
  https://support.somnomedics.de/wp-content/uploads/2020/02/Manual_DOMINO_3.0.0_eng_Rev_1_2019_10_29_WAM-1.pdf
- Kennisbank "How to export raw data as EDF-file?":
  https://support.somnomedics.de/?kbx_knowledgebase=how-to-export-raw-data-as-edf-file-wie-konnen-rohdaten-als-edf-datei-exportiert-werden
- Kennisbank "Exporting the body position as EDF, what are the values standing for?":
  https://support.somnomedics.de/?kbx_knowledgebase=exporting-the-body-position-as-edf-what-are-the-values-standing-for
- Kennisbank "How import EDF files from a foreign device?":
  https://support.somnomedics.de/?kbx_knowledgebase=how-import-edf-files-from-a-foreign-device
- Productpagina DOMINO (export Excel/SPSS/EDF+/ASCII, import EDF+):
  https://somnomedics.de/en/solutions/sleep_diagnostics/diagnostic_software/domino/
- Snooz Toolbox, "Convert Annotations from DOMINOScreen" (ASCII-bestandsvorm):
  https://snooz-toolbox-documentation.readthedocs.io/1.1.0/user_guide/tools/Preprocessing/DominoScreen_Converter.html
- SpiSOP downloads (conversiescripts "Somnomedics DOMINO export (txt)"): https://www.spisop.org/downloads/
- EDFbrowser manual (annotaties importeren/exporteren, hypnogram-labelmapping):
  https://www.teuniz.net/edfbrowser/EDFbrowser%20manual.html
