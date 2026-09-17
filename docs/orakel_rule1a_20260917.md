# Orakel-decompositie van de Rule-1A-arousaltak — verslag

*17 september 2026. Pre-registratie: `orakel_rule1a_preregistratie_20260917.md`
(geschreven vóór de eerste run). Harnas: `scripts/orakel_rule1a.py`.
Keten: psgscoring 0.34.2, git `8dac397`, `aasm_v3_rec`. Ruwe uitvoer
(7 MB, per nacht per arm de eventlijsten): `/srv/CODE/docs/orakel_rule1a_20260917/`.*

## Vraag en opzet

De arousaltak van Rule 1A is op 29-08 weerlegd (MESA n=150 vs `aasm15`:
F1 0,438→0,382, bias −5,26→+8,01). Twee oorzaken bleven open: onze
arousals (fout-positieven op koppelplekken) of de koppelregel zelf. Drie
armen, alle in-pipeline op dezelfde raw/hypno: **A** tak uit (productie),
**B** tak aan met onze arousals, **C** tak aan met de
*referentie*-arousals (PSG-IPA: vaste EEG-arousalannotatie; MESA: NSRR
`Arousals|Arousals`, via `run_pneumo_analysis(arousal_events=…)`).

Getrouwheid: arm A reproduceert op PSG-IPA de paper-v31-AHI's
bit-identiek (8,1 / 9,3 / 53,8 / 4,3 / 11,0; bias +1,69, F1 0,349) en
op MESA de 29-08-cijfers (F1 0,438, precisie 0,520).

## PSG-IPA, n=5 (descriptief)

| arm | herstellingen | herstelprecisie (gepoold) | bias | F1-mediaan |
|---|---|---|---|---|
| A | 0 | — | +1,69 | 0,349 |
| B | 16 | 1/16 = 0,06 | +2,19 | 0,342 |
| C | 5 | 1/5 = 0,20 | +1,85 | 0,353 |

Op 75 geteste kandidaten koppelen onze arousals 20 keer, de menselijke 5
keer. Te klein om te beslissen; MESA beslist.

## MESA, n=149 van 150 (één nacht zonder `aasm15`-events), referentie `aasm15`

| arm | F1 med | precisie | recall | bias (gem.) | herstellingen | **R1 herstelprecisie** (gepoold / med) | R3 gat-recall | ernst = ref |
|---|---|---|---|---|---|---|---|---|
| A | 0,438 | 0,520 | 0,412 | −5,61 | 0 | — | 0,246 | 87 |
| B | 0,447 | 0,508 | 0,450 | −4,03 | 1372 | **0,254** / 0,185 | 0,329 | 81 |
| C | 0,461 | 0,523 | 0,463 | −4,38 | 1082 | **0,436** / 0,470 | 0,366 | 84 |

R2 gepaarde ΔF1: **C−A mediaan +0,005, gemiddeld +0,024, beter op 89/149,
Wilcoxon p = 1,8·10⁻⁹**; B−A mediaan 0,000, gemiddeld +0,003, 68/149,
p = 0,20 (neutraal).

Per AHI-tertiel (ref `aasm15`):

| tertiel | n | B: ΔF1 (beter) | B: bias A→B | B: R1 med | C: ΔF1 (beter) | C: bias A→C | C: R1 med |
|---|---|---|---|---|---|---|---|
| T1 0,3–13,0 | 49 | −0,004 (16, p=0,26) | +2,5 → +4,8 | 0,06 | +0,046 (26, p=0,003) | +2,5 → +4,0 | 0,19 |
| T2 13,3–28,2 | 50 | +0,009 (27, p=0,05) | −4,0 → −2,2 | 0,29 | +0,020 (35, p=4·10⁻⁶) | −4,0 → −2,4 | 0,45 |
| T3 28,5–88,9 | 50 | +0,003 (25, p=0,002) | −15,2 → −14,5 | 0,41 | +0,006 (28, p=3·10⁻⁶) | −15,2 → −14,6 | 0,60 |

Secundaire referentie `desat3_all` (desaturatie-only, kán arousal-only
hypopneus niet crediteren): B en C verliezen daar per constructie F1
(−0,021 / −0,016, beide p < 10⁻¹⁵) en 95 % van de herstellingen matcht er
niets — dat bevestigt dat de herstellingen inderdaad de arousal-only-klasse
zijn, niet gemiste desaturatie-events.

## Wat de orakel-koppelingen zijn

Elke C-herstelling (1082/1082) heeft een referentie-arousal binnen
[onset−5 s, einde+15 s]: 55 % begint *tijdens* het event, 37 % binnen 5 s
na het einde, 7 % tussen 5 en 15 s. De koppeling zelf is dus getrouw; het
venster is niet het probleem (bevestigt 21-08). Herstelde events zijn lang:
mediaan 25 s, 90e percentiel 50 s.

Gevoeligheid (post-hoc, gelabeld): met de referentie herbouwd op een
arousalvenster van 15 s i.p.v. 5 s wordt R1(C) 0,508 en ΔF1(C−A) +0,028
(93/149, p = 1,3·10⁻¹²); R1(B) 0,304. Het referentievenster verklaart dus
maar een deel: ook dan is de helft van de orakel-herstellingen géén
referentie-hypopneu.

## Besluit volgens de vooraf vastgelegde regel

R1(C) = 0,436 < 0,50 → **"koppeling te toegeeflijk"**. Met perfecte
arousals is 56 % van wat de tak herstelt geen referentie-event — terwijl
de arousal er wél echt is. De toegeeflijkheid zit dus niet in venster of
gap (die zijn getrouw), maar in de **kandidaat-eligibility**: elke
`no_desaturation`-afwijzing van ≥30 %/≥10 s mag terugkomen, ook lange,
zwakke debietdalingen die een mens ook mét arousal niet scoort.

Twee nuances horen erbij. (1) Het orakel *wint* wel: ΔF1 +0,024 op 89/149,
p = 1,8·10⁻⁹, in elk tertiel positief, en het dicht het structurele gat
van 25 % naar 37 % recall — de regelhelft is echt en de koppelregel heeft
waarde, alleen met te veel bijvangst. (2) Onze eigen arousals (B) zijn op
de huidige keten niet meer de ramp van 29-08 (F1 0,447 tegen 0,382 toen;
bias −4,0 tegen +8,0): de keten sinds 0.32.0 (union, 0,70 + 10 s,
re-ranker) én de eligibility-filter van `fa5dd69` deden hun werk. B blijft
F1-neutraal en blaast T1 op (bias +2,5→+4,8, R1 0,06); hij blijft uit.

## Wat hieruit volgt (geen besluit; ontwerp voor een volgende meting)

Niet: het venster of de gap aanpassen. Wél: een **strengere kandidaatpoort
voor arousal-kwalificatie** — bijv. een gegradeerde debiet/duur-term
(`p_scored`-achtig) waarbij een hypopneu alleen via een arousal mag
kwalificeren als de debietdaling zelf overtuigend is, en een duurplafond
op herstelde kandidaten. Meten met exact dit harnas (C als plafond: als de
poort onder het orakel geen R1 ≥ 0,60 haalt, is de poort fout, niet de
arousals) en pas daarna B opnieuw. Arousalprecisie op koppelplekken blijft
de tweede as: B→C is 0,25→0,44.

## Rekenkundig

150 nachten × 3 armen in 95 min: 20 workers × 1 BLAS-thread; RAM-dieptepunt
39 GB vrij van 151; **piek 82 °C** (x86_pkg_temp) bij aanhoudende
vollast — dicht bij de 85 °C-bewaker, die niet hoefde in te grijpen. Voor
runs van deze duur is 20 workers het plafond op deze koeling, niet 24.
De n150-set overlapt voor 100 nachten met later geregistreerde
afleidingssets; hier is niets afgesteld; registratie in
`gebruikte_mesa_ids.txt`.
