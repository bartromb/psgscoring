# Varianten op breath (en ernaast) — denkstuk, vooruitlopend op de avondmetingen

*2026-09-03, geschreven terwijl de 450-run en rec-tegen-breath draaien. Dit is
een rangschikking van opties, geen besluit en geen bouwplan; elke variant hier
volgt de huisregel — bouwen achter een vlag, meten op twee cohorten, vooraf
vastgelegde regel.*

## Waar breath nu staat

`aasm_v3_breath` verschilt van `rec` in één kern: `HYPOPNEA_DETECTOR:
envelope → breath_graded`. De ademteug is het atoom, kalibratie per patiënt in
twee passages (incl. eigen SpO₂-vertraging via kruiscorrelatie), gegradeerde
AASM-predicaten, en één strengheidsas.

Bewijs tot nu toe: MESA n=150 op 0.17.0 event-F1 **+0,029** (p=6,8e-8) bij
vergelijkbare bias; PSG-IPA op 0.32.0 **5/5 binnen de scoordersspreiding**
(|bias| 0,74 tegen 1,76) en 5/5 juiste ernstklassen. De MESA-herhaling op
0.32.0 draait vanavond.

## De varianten, gerangschikt

### 1. `breath` op strictness 0,30 — de knop bestaat al en heeft al gewonnen

De strengheidsas is precies punt 5 van het breath-ontwerp, default 0,50. De
herijking (zie geheugennotitie hypopneu-strictness) mat: **0,30 haalt alle
drie de vooraf gestelde criteria** — validatie +0,0346 op 26/30, bias
−3,28 → −0,05 — maar is nooit uitgerold, en het adaptieve alternatief is op
zijn eigen plafond weerlegd (orakel 0,607 tegen vast-0,30 0,588).

Als breath vanavond wint, is dít de eerstvolgende meting: `breath@0,30` tegen
`breath@0,50`, zelfde opzet als vanavond. Kosten: nul bouw, één run.
Risico: de 0,30-meting is van vóór een reeks releases; herbevestigen, niet
aannemen.

### 2. p_scored kalibreren → een verwachtingswaarde-AHI

`p_scored` ordent goed maar ligt ~33 procentpunt te hoog t.o.v. de werkelijke
scorerfractie (r=0,194; band 0,90+ → 0,58). PSG-IPA levert per event een
doelwaarde: de fractie van twaalf scoorders die hem markeert.

De variant: isotone kalibratie leave-one-recording-out van p_scored naar
scorerfractie, en dan de **AHI als som van gekalibreerde kansen** in plaats
van als telling van binaire events. Dat richt zich rechtstreeks op het doel
"binnen de menselijke spreiding" en levert er gratis een onzekerheidsband bij.

Kosten: matig (kalibratiescript bestaat deels: `calibrate_on_multiscorer.py`).
Risico's: n=5 opnames voor de kalibratie; en op MESA (één scoorder) is alleen
het niveau te valideren, niet de fractie. Dit is de meest onderzoeksachtige
optie en de enige die iets fundamenteel nieuws aan het rapport toevoegt.

### 3. Conditionele apneu-gradering via de CSR-detector — antwoord op de terugrol

De s=0,25-omkering is basiskansafhankelijkheid: winst waar centrale apneus
frequent zijn (periodieke ademhaling), schade waar ze zeldzaam zijn. Er ZIT
al een Cheyne-Stokes-detector in `ancillary.py`. De variant: gradering alleen
in opnames (of segmenten) waar CSR/periodieke ademhaling gedetecteerd is.

Tegenargument dat serieus genomen moet worden: een adaptief werkpunt is bij de
hypopneu-strictness al eens op zijn plafond weerlegd. Maar dit is geen continu
adaptief werkpunt — het is een binaire contextpoort op een al bestaande
detector, en het mechanisme (basiskans) is gemeten, niet vermoed. De
450-stratificatie van vanavond zegt of het plafond van deze poort de moeite
waard is: winst-op-CSR-opnames × prevalentie.

### 4. `breath_dual` herijken wáár de thermistorpoort doorlaat

Op 0.17.0 had breath_dual de kleinste MESA-bias (−2,34). De duale as is alleen
te beoordelen op opnames met een doorgelaten thermistor (~33 % op MESA; de
poort is de goede sensor om de verkeerde reden). Pas relevant als breath wint;
dan één gestratificeerde run: poort-open-opnames apart.

### 5. Geen scoringswijziging: de spreidingsband in het rapport

`expected_scorer_agreement` (ingebouwd) en de twee profielen samen kunnen nu
al leveren wat een clinicus eigenlijk wil weten: "rec zegt 9,3, breath zegt
4,9, twaalf menselijke scoorders zouden 1,7–6,8 zeggen". Eén regel in het
PDF-rapport, geen enkele detector aangeraakt. Dit kan ongeacht alle
bovenstaande beslissingen en maakt de onzekerheid zichtbaar in plaats van
haar weg te middelen.

## Wat ik NIET voorstel

* **Envelope-varianten** (chunked, rectify, decimated): gemeten en niet
  gepromoveerd; rectify repliceerde niet over cohorten.
* **Adaptieve strictness**: op zijn orakelplafond weerlegd.
* **Duale as als default**: de thermistorpoort keurt de meeste montages af en
  wáár hij doorlaat mat de duale as slechter (v38-tijdperk); eerst variant 4.

## Volgorde als vanavond beide verwachtingen uitkomen

1. terugrolbeslissing s=0,25 (met de 450-stratificatie in de hand, en
   variant 3 als constructief alternatief naast kaal terugrollen);
2. breath-beslissing (beide cohorten naast elkaar, aan de gebruiker);
3. bij een breath-keuze: meteen variant 1 meten vóór enige uitrol;
4. variant 5 kan parallel, want hij verandert geen scoring.
