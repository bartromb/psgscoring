# Schaalmeting arousal-U-Net — uitslag (preregistratie 25-10-2026; meting nacht 25/26-10)

Preregistratie `docs/arousal_unet_schaal_preregistratie_20261025.md`, sha256
`0c7faf2723f5b555…` — gelogd bij de start van de keten (25-10 20:37:50) en ongewijzigd bij de
evaluatie. Keten `bench/eeg/unet_scale/run_chain.sh` → `train_scale.py` (seed 20261025, git
58c3e06 + de ongecommitte werkboom van 25-10; de bench roept de arousaldetector van de bibliotheek
niet aan, alleen `bench/evaluate.py` en de 10 s-samenvoeging), `ALLES_KLAAR` 26-10 03:58:02 na
440 min. Evaluatie `bench/eeg/unet_scale/evaluate_scale.py` →
`/srv/CODE/docs/arousal_unet_schaal_20261025/evaluatie.{md,json}`; ruwe uitvoer buiten git in
die map (per arm `model_best.pt`, `tau.json`, `train_log.json`; per testset `_ref.csv`,
`_pred.csv`, `_grid.json`, `infer_log.jsonl`; `laadfouten.json`; `chain.log`; `thermal.log`).

**Uitslag in één zin.** De arm **MS** (1 262 MESA- + 1 235 SHHS1-nachten, EEG/EOG/kin-EMG)
haalt de volledige vooraf vastgelegde regel: op het nooit geziene SHHS1-150b stijgt de mediane
event-F1 van 0,670 naar 0,739 (gepaard +0,057, beter op 134 van 148 nachten, p = 4·10⁻²⁴), op
MESA-76 +0,011 (p = 0,005), op PSG-IPA op 4 van 5 nachten niet lager. M (alleen vier keer meer
MESA) en MSF (MS + flowkanaal) halen de primaire en de MESA-voorwaarde maar niet de
PSG-IPA-voorwaarde (3/5 en 2/5). **Het flowkanaal voegt niets toe**: MSF − MS is 0,000 op
SHHS1 (73 beter / 73 slechter, p = 0,73). Volgens de prereg is MS daarmee **kandidaat voor een
`unet_v2`-gewicht**; inbouw, default en werkpunt zijn Barts beslissing — niet in deze meting.

## Wat er precies gemeten is

- **Trainingspool**: 1 278 + 1 278 geselecteerde nachten (`select_ids.py`, geregistreerd in
  `gebruikte_mesa_ids.txt` sectie 2 en `gebruikte_shhs_ids.txt` sectie 3); 59 vielen uit bij
  het laden (45 "te weinig slaap of geen arousals", 14 SHHS1-nachten zonder flowkanaal — de
  lader eist flow voor alle armen, zodat M, MS en MSF dezelfde nachten zien) → MESA 1 262,
  SHHS1 1 235. Validatie 40 + 40 (val-80). Flow in de training: MESA `Pres` 1 262 / `Therm`
  1 253, SHHS1 alleen thermistortype 1 678 (per venster willekeurig gekozen, zie prereg).
- **Armen**: M = 1 262 MESA, 3 ingangen, ≤ 50 epochs (early stop na 43, beste 32); MS = M +
  1 235 SHHS1, 3 ingangen, ≤ 60 epochs (**liep tot de grens**, beste 56); MSF = MS + flow, 4
  ingangen, ≤ 60 (**liep tot de grens**, beste 54). Zelfde architectuur als `unet_v1`
  (4,58 M parameters; 600 stappen × 32 vensters van 600 s per epoch, 76–80 s/epoch op de A4000).
- **τ per model uit val-80** (F1-optimaal / tellingsneutraal): bevroren 0,25 / 0,20 (val-F1
  0,683), M 0,25 / 0,20 (0,711), MS 0,30 / 0,20 (0,730), MSF 0,30 / 0,20 (0,732). Het bevroren
  model krijgt twee extra rijen: τ 0,20 (bench-keuze 27-09) en **τ 0,35 (bibliotheekwerkpunt)**.
- **Testsets** (nooit in training of validatie): SHHS1-150b 150 verse nachten (148 geladen;
  18 639 NSRR-arousals), MESA-76 (12 009), PSG-IPA n = 5 (596; referentie = scoorder 1 van de
  bench, al gezien bij de keuze van `unet_v1`, dus bevestiging, geen bewijs).
- **Matcher**: projectmatcher IoU 0,20 (`validate_psgipa.LEGACY_MATCHER`) na de
  10 s-samenvoeging van de bibliotheek (interval uit `aasm_v3_rec`), gepaard per nacht;
  elke-overlap ("any") ter vergelijking. REM/NREM op het NSRR-hypnogram; count-ratio = eigen
  events / referentie per nacht; bias = eigen index − referentie-index (/u TST).
- **Bewakers**: CPU-pakket max 64 °C (p95 60 °C), GPU max 85 °C, 0 thermische pauzes, RSS-piek
  69,3 GB (naast de hPG-simulatie met 3 workers; MemAvailable bleef > 100 GB).

## 1. SHHS1-150b — 148 nachten, beslissend

| variant | F1 med | F1 gepoold | sens | PPV | F1 any med | REM F1 | NREM F1 | count-ratio med | bias /u |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bevroren τ 0,25 | 0,670 | 0,674 | 0,666 | 0,683 | 0,681 | 0,615 | 0,737 | 0,97 | −0,75 |
| bevroren τ 0,35 (bibliotheek) | 0,658 | 0,673 | 0,612 | 0,747 | 0,675 | 0,646 | 0,736 | 0,81 | −4,03 |
| M τ 0,25 | 0,693 | 0,712 | 0,674 | 0,755 | 0,699 | 0,687 | 0,768 | 0,89 | −2,47 |
| **MS τ 0,30** | **0,739** | **0,742** | 0,711 | 0,777 | **0,739** | **0,761** | **0,794** | 0,92 | −1,94 |
| MSF τ 0,30 | 0,733 | 0,742 | 0,698 | 0,792 | 0,736 | 0,759 | 0,795 | 0,86 | −2,65 |
| MS τ 0,20 (neutraal) | 0,731 | 0,736 | 0,754 | 0,718 | 0,737 | 0,731 | 0,783 | 1,05 | +0,97 |

| paar | ΔF1 med | gem | beter/slechter/gelijk | Wilcoxon p | \|cr−1\| med arm vs ref | tertiel laag/midden/hoog |
|---|---:|---:|---|---:|---|---|
| M − bevroren | +0,025 | +0,038 | 115/32/1 | 1,8·10⁻¹³ | 0,157 vs 0,187 | +0,046 / +0,019 / +0,017 |
| **MS − bevroren** | **+0,057** | +0,070 | **134/13/1** | **4,4·10⁻²⁴** | 0,160 vs 0,187 | +0,071 / +0,056 / +0,045 |
| MSF − bevroren | +0,053 | +0,071 | 137/10/1 | 2,7·10⁻²⁴ | 0,163 vs 0,187 | +0,066 / +0,051 / +0,040 |
| **MSF − MS (flow)** | **0,000** | +0,001 | **73/73/2** | **0,73** | 0,163 vs 0,160 | +0,007 / +0,002 / −0,002 |
| MS − bevroren τ 0,35 | +0,056 | +0,075 | 133/14/1 | 4,4·10⁻²³ | — | — |

Per-nacht-F1 p10/p50/p90: bevroren 0,545 / 0,670 / 0,786 → MS 0,610 / 0,739 / 0,823. Grootste
verliezen van MS: −0,035, −0,027, −0,025 (geen nacht verliest meer dan 0,04). De winst zit
in alle tertielen (43/6, 44/5, 47/2 beter/slechter) en het meest in **REM**: sens 0,763 → 0,817
bij PPV 0,515 → 0,713 (REM-F1 0,615 → 0,761) — precies de zwakke plek van het bevroren model
op SHHS1 (27-09: REM-PPV 0,58). NREM 0,737 → 0,794.

**Primaire regel (per arm tegen bevroren):** mediane ΔF1 ≥ +0,02 → M +0,025 JA, MS +0,057 JA,
MSF +0,053 JA; p < 0,01 → 3× JA; ≥ 60 % beter → 78 % / 91 % / 93 % JA; mediane |count-ratio − 1|
≤ bevroren + 0,05 → 0,157 / 0,160 / 0,163 tegen 0,187 JA (alle drie beter dan bevroren).

## 2. MESA-76 — tweede replicatie

| variant | F1 med | F1 gepoold | sens | PPV | F1 any med | REM F1 | NREM F1 | count-ratio med | bias /u |
|---|---|---|---|---|---|---|---|---|---|
| bevroren τ 0,25 | 0,683 | 0,699 | 0,648 | 0,759 | 0,701 | 0,699 | 0,760 | 0,83 | −4,33 |
| bevroren τ 0,35 (bibliotheek) | 0,670 | 0,687 | 0,597 | 0,809 | 0,692 | 0,698 | 0,757 | 0,72 | −7,37 |
| M τ 0,25 | 0,695 | 0,707 | 0,660 | 0,760 | 0,715 | 0,708 | 0,771 | 0,84 | −3,96 |
| **MS τ 0,30** | 0,699 | 0,708 | 0,647 | 0,783 | 0,712 | 0,708 | 0,776 | 0,80 | −5,03 |
| MSF τ 0,30 | 0,702 | 0,711 | 0,648 | 0,789 | 0,720 | 0,720 | 0,778 | 0,77 | −5,19 |
| MS τ 0,20 (neutraal) | 0,699 | 0,709 | 0,690 | 0,729 | 0,727 | 0,694 | 0,766 | 0,91 | −1,90 |

| paar | ΔF1 med | gem | beter/slechter/gelijk | Wilcoxon p | p eenzijdig "slechter" | tertiel laag/midden/hoog |
|---|---:|---:|---|---:|---:|---|
| M − bevroren | +0,010 | +0,010 | 48/26/2 | 0,0029 | 0,999 | +0,011 / +0,011 / +0,006 |
| **MS − bevroren** | **+0,011** | +0,009 | **50/26/0** | **0,0052** | **0,997** | +0,014 / +0,009 / +0,009 |
| MSF − bevroren | +0,013 | +0,012 | 52/24/0 | 0,0053 | 0,997 | +0,018 / +0,011 / +0,011 |
| MSF − MS (flow) | +0,003 | +0,003 | 41/33/2 | 0,28 | 0,86 | +0,008 / −0,002 / +0,005 |
| MS − bevroren τ 0,35 | +0,017 | +0,020 | 59/14/3 | 4,7·10⁻⁷ | — | — |

Grootste verliezen van MS: −0,087 (`mesa-sleep-6123`, 24 referentie-arousals), −0,077, −0,061.
MESA stond al in de training van het bevroren model (317 nachten); vier keer meer MESA (M) en een
tweede cohort (MS) brengen hier +0,01 — de verbetering op MESA is klein maar systematisch.
**Secundaire regel:** mediane ΔF1 ≥ 0 → JA (alle drie); eenzijdig niet slechter p ≥ 0,05 → JA.

## 3. PSG-IPA n = 5 — bevestiging op een geziene set

| nacht | n ref | bevroren τ 0,25 | bevroren τ 0,35 | M | **MS** | MSF |
|---|---:|---:|---:|---:|---:|---:|
| SN1 | 37 | 0,753 | 0,709 | 0,639 | 0,667 | 0,633 |
| SN2 | 96 | 0,654 | 0,678 | 0,670 | **0,678** | 0,704 |
| SN3 | 241 | 0,873 | 0,883 | 0,863 | **0,876** | 0,871 |
| SN4 | 58 | 0,474 | 0,471 | 0,488 | **0,482** | 0,471 |
| SN5 | 164 | 0,765 | 0,713 | 0,821 | **0,810** | 0,791 |
| gepoold | 596 | 0,749 | 0,745 | 0,757 | **0,763** | 0,759 |

Plafond PSG-IPA 0,679 (mens-tegen-mens, andere export, niet like-for-like). MS tegen bevroren:
**niet lager op 4 van 5** (mediane ΔF1 +0,008; alleen SN1, de nacht met de minste arousals,
verliest 0,086); M 3/5, MSF 2/5; MSF − MS 1/4 (−0,011). Count-ratio mediaan bevroren 1,14 → MS
1,08 (bias +2,84 → +2,37 /u); REM-F1 0,629 → 0,657, NREM 0,764 → 0,778. Met n = 5 is geen
p-waarde zinvol; dit is de vooraf afgesproken "niet slechter"-poort, en die haalt alleen MS.

## 4. Beslisregel (vooraf vastgelegd) — uitslag

| arm | primair SHHS1-150b (4 voorwaarden) | secundair MESA-76 | secundair PSG-IPA ≥ 4/5 | **geslaagd** |
|---|---|---|---|---|
| M | JA (+0,025; p 2·10⁻¹³; 78 %; cr beter) | JA (+0,010; p₁ 0,999) | **NEE (3/5)** | nee |
| **MS** | **JA** (+0,057; p 4·10⁻²⁴; 91 %; cr beter) | **JA** (+0,011; p₁ 0,997) | **JA (4/5)** | **JA** |
| MSF | JA (+0,053; p 3·10⁻²⁴; 93 %; cr beter) | JA (+0,013; p₁ 0,997) | **NEE (2/5)** | nee |
| flow: MSF − MS | ΔF1 0,000 < +0,01 **NEE**; p 0,73 **NEE**; 49 % **NEE**; cr JA | — | — | **nee** |

`evaluatie.json["beslisregel"]` bevat dezelfde uitkomst machinaal.

## 5. Wat de drie armen uit elkaar halen

- **Schaal alleen** (M: 317 → 1 262 MESA-nachten, zelfde cohort): +0,025 op SHHS1, +0,010 op
  MESA. Vier keer meer nachten van hetzelfde cohort is een bescheiden winst.
- **Tweede cohort** (MS − M): nog eens +0,032 op SHHS1 (dat cohort zit nu in de training —
  domeinaanpassing, geen zuivere schaalwinst), +0,001 op MESA, +0,006 gepoold op PSG-IPA. Voor
  klinische PSG's die in geen van beide cohorten zitten is de verwachte winst dus kleiner dan de
  SHHS1-kop van +0,057 suggereert; PSG-IPA (Somnomedics-achtige montage, scoorder 1) laat
  +0,008 mediaan zien, met één verliezende nacht.
- **Flow** (MSF − MS): niets, op drie cohorten, en op PSG-IPA eerder negatief. De vierde ingang
  zou de bibliotheek voor de arousals afhankelijk maken van een flowkanaal; dat is nu niet
  gerechtvaardigd. De Pitkänen-aanname "nasale druk helpt de arousaldetector" repliceert in deze
  opstelling niet (SHHS1 heeft alleen thermistortype flow; op MESA-76, mét druk, is het +0,003).
- **τ ruilt telling tegen precisie, zoals op 27-09.** F1 van MS is vlak over τ 0,20–0,30 (SHHS1
  0,731–0,739, MESA 0,699); τ 0,30 telt 0,92 / 0,80 / 1,08 van de referentie (SHHS1 / MESA /
  PSG-IPA), τ 0,20 telt 1,05 / 0,91 / 1,15. Het werkpunt hoort bij het gewicht (prereg 27-09)
  en is een keuze, geen meting; het bibliotheekwerkpunt 0,35 ligt voor MS buiten het vlakke
  stuk (val-F1 0,725 tegen 0,730) en zou met een `unet_v2` opnieuw gekozen moeten worden.

## 6. Kanttekeningen

- **MS en MSF liepen tot de epochgrens (60)**, beste epoch 56 en 54; M stopte vroeg (43, beste
  32). Beide grote armen waren bij de grens nog niet uitgeleerd; de val-F1-curve vlakt af maar
  daalt niet. Een langere training kan nog iets opleveren — dat is een nieuwe meting, geen
  correctie op deze.
- **Het bevroren model is op τ 0,25 vergeleken** (zijn eigen val-τ, conform de prereg), niet op
  het bibliotheekwerkpunt 0,35; tegen 0,35 is de MS-winst groter (+0,056 SHHS1, +0,017 MESA).
- **Twee SHHS1-testnachten vielen uit bij het laden** (`shhs1-202089`, `shhs1-205030`; zelfde
  lader als de training, de reden staat niet apart in `chain.log`); n = 148 in plaats van 150.
  Geen nacht uit de pool is bijgeladen.
- **PSG-IPA** blijft een geziene set van 5 met één scoorder als referentie; het 4/5-criterium is
  grof. De nacht die verliest (SN1) heeft 37 referentie-arousals: één gemiste of extra arousal
  weegt daar 0,03.
- **Bench-inferentie in bfloat16-autocast** op de GPU; de ONNX-export is float32 op de CPU.
  Dat verklaart kleine verschillen tussen deze cijfers en een latere bibliotheekmeting; de
  export zelf wijkt < 2·10⁻⁷ af van torch-float32.
- **Niet gemeten** (zoals vooraf afgebakend): doorwerking op RERA/RDI, hypopneu-koppeling,
  PLM-associatie, arousal-index in de klinische keten, CPU-kost (identiek aan v1 te verwachten:
  zelfde architectuur, zelfde drie ingangen).

## 7. Gevolg volgens de preregistratie

MS is **kandidaat voor `unet_v2`**. Klaargezet, buiten de bibliotheek:
`/srv/CODE/docs/arousal_unet_schaal_20261025/MS/arousal_unet_MS_kandidaat.onnx` (18,3 MB, opset
17, sigmoid in de graaf, sha256 `9a084f8814a517f5…`; bron `model_best.pt` sha256
`e75bee7a409e18af…`, epoch 56; metadata in `arousal_unet_MS_kandidaat.json`, exporteur
`bench/eeg/unet_scale/export_onnx.py`; max. afwijking tegen torch 1,6·10⁻⁷ op 12,8 min en 2,1 u).

Wat daarna nog nodig is en **niet in deze meting zit**: (1) Barts beslissing over inbouw als
tweede gewicht naast `unet_v1` (zelfde terugval, sha256-wacht, re-ranker inert), over de default
(`unet_v1` blijft tot dan de standaard op de klinische profielen) en over het werkpunt (0,25 of
0,30 voor MS; 0,35 is voor dit gewicht suboptimaal); (2) de klinische aan/uit-controle op eigen
nachten (arousal-index, RDI, respiratoir byte-gelijk) zoals bij 0.35.0; (3) de doorwerkingsmeting
zoals 09-10 als het de default wordt. Het bevroren `unet_v1` blijft tot die beslissingen het
gewicht in de bibliotheek. De 1 278 + 1 278 trainingsnachten gelden voortaan als
arousal-getraind (registers bijgewerkt op 25-10); SHHS1-150b en MESA-76 zijn nu voor een tweede
keer als testset gebruikt en zijn voor een eventuele τ-keuze van `unet_v2` **afstelset**, geen
verse cohort meer — SHHS1 heeft nog ≈ 4 200 ongebruikte nachten.
