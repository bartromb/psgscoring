# Preregistratie — wat doet de arousal-classifier met de RDI op de RERA-dragende profielen?

*23 augustus 2026, ná een geslaagde pre-flight en vóór de uitkomst.*

## De openstaande beslissing

De classifier staat sinds 22-08 **uit** op de vier profielen waar arousals de
RDI dragen (`aasm_v3_breath`, `aasm_v3_prob`, + duals). Reden toen: hij
verschoof de RDI-ernstklasse op 28 % van de opnames en er is geen
RERA-referentie om te bepalen welke RDI juister is.

Sindsdien is er iets veranderd: het **werkpunt** is herijkt van 0,60 naar 0,80,
en 0,60 was aantoonbaar gedomineerd. De vraag is dus opnieuw open — maar nu met
een classifier die op een ander punt draait dan toen de beslissing viel.

Deze opnames draaien op deze profielen ook in de kliniek; het is de configuratie
waar de RDI aan hangt.

## Waarom de vorige poging niets mat

Twee oorzaken, allebei van mij:

1. `aasm_v3_breath` heeft `arousal_lgbm=False`, dus de drempel deed daar niets;
2. de pipeline las `AROUSAL_LGBM` **zonder env-override** — als enige
   arousalvlag — zodat de armen niet te scheiden waren.

Resultaat: 30/30 identiek, wat eruitzag als "geen effect". De override is nu
toegevoegd, met een test die faalt als de armen gelijk uitkomen.

**Pre-flight gedaan** op `mesa-sleep-0407` vóór deze preregistratie werd
afgerond: classifier uit → AHI 66,5 / RDI 67,0 / 42 arousals; aan → AHI 64,9 /
RDI 65,4 / 30 arousals. De armen verschillen dus aantoonbaar.

## Opzet

- MESA n=30, zaad 20260824, gepaard.
- Profiel `aasm_v3_breath` — RERA-dragend, en de klinische werkpaard.
- **Artefact-epochs meegegeven** (`PSGSCORING_HARNESS_ARTIFACT_EPOCHS=1`),
  zodat de meting draait zoals productie draait. Dat was tot vandaag niet zo.
- Overige stand = uitgerold: arousalstap negeert de lijst, drempel 0,80.

| arm | classifier |
|---|---|
| A | uit (huidige stand op deze profielen) |
| B | aan, werkpunt 0,80 |

## Karakterisering, geen slaag/zak

Er is geen RERA-referentie, dus "juister" is niet meetbaar
([[reference_no_rera_reference]]). Wat gemeten wordt is de **omvang**: AHI,
RDI, arousal-index, RERA-index, en de gepaarde verschuivingen.

## Meldgrens — vooraf

Verschuift de **RDI-ernstklasse** op meer dan een kwart van de opnames, dan leg
ik dat als apart punt voor in plaats van het in een verslag te vermelden.
Dezelfde grens en dezelfde reden als bij de vorige RDI-karakterisering, die
28 % gaf.

Ik noteer de AHI-ernstklasse er apart bij: op dit profiel bevestigen arousals
ook hypopneus, dus de AHI beweegt mee — de pre-flight liet 66,5 → 64,9 zien.
Dat maakt dit géén zuivere RDI-vraag, en dat hoort vooraf te staan.

## Wat dit niet uitwijst

Welke arm de betere RDI geeft. Alleen hoe groot het verschil is, zodat de
beslissing met dat cijfer erbij genomen wordt in plaats van erna.

---

# Uitkomst — 23 augustus 2026

**De meldgrens is gehaald: de RDI-ernstklasse verschuift op 37 % van de
opnames.** Apart voorgelegd.

Gepaard over 30 MESA-opnames, `aasm_v3_breath`, drempel 0,80, artefact-epochs
meegegeven zoals productie.

| | A: classifier uit (huidig) | B: aan | verschil |
|---|---:|---:|---:|
| AHI mediaan | 19,45 | 19,05 | −0,40 |
| **RDI mediaan** | **34,25** | **28,70** | **−5,55** |
| arousals mediaan | 212 | 107 | −105 |
| arousal-index mediaan | 35,60 | 19,50 | −16,10 |
| RERA-index mediaan | 14,85 | 6,80 | −8,05 |
| **event-F1 mediaan** | **0,44** | **0,48** | **+0,04** |
| AHI-bias mediaan | −2,09 | −3,28 | −1,18 |

Gepaarde ΔRDI: mediaan **−9,25/u**, omlaag op 26/30. Gepaarde ΔF1 **+0,0090**,
beter op 23/30, Wilcoxon **p = 5,5·10⁻⁴**.

| | verschuift |
|---|---|
| **RDI-ernstklasse** | **11/30 = 37 %** |
| AHI-ernstklasse | 4/30 = 13 % |

## De arousaltelling heeft hier WEL een referentie

Anders dan de RDI. Tegen de MESA-annotatie op dezelfde 30 opnames:

| | mediaan | ratio t.o.v. referentie |
|---|---:|---:|
| MESA-referentie (menselijk) | 128 | 1,00 |
| A: classifier uit (huidig) | 212 | **1,47** |
| B: classifier aan (0,80) | 107 | **0,83** |

Afwijking van een zuivere telling: **0,58 → 0,26**. B ligt dichter bij de
referentie op **22 van 30** opnames.

Dat is het beslissende gegeven dat bij de vorige RDI-karakterisering ontbrak.
De huidige stand overdetecteert arousals met bijna de helft; met de classifier
telt hij 17 % te weinig. Beide zijn scheef, maar B minder dan half zo scheef.

## Wat dit wel en niet zegt

**Wel:** de arousaltelling wordt aantoonbaar beter, en de respiratoire event-F1
ook (+0,009, p = 5,5e-04, 23/30).

**Niet:** dat de resulterende RDI juister is. Er is geen RERA-referentie, dus
een RDI die 9,25/u lager uitkomt is niet als correctie te boeken — alleen als
verschuiving. En de AHI-bias verslechtert licht (−2,09 → −3,28), want op dit
profiel bevestigen arousals ook hypopneus.

## Waarom de vorige poging niets mat

`aasm_v3_breath` heeft `arousal_lgbm=False`, én de pipeline las `AROUSAL_LGBM`
als enige arousalvlag zonder env-override. De armen waren niet te scheiden:
30/30 identiek, wat eruitzag als "geen effect". Override toegevoegd met een
test die faalt als de armen gelijk uitkomen, en een pre-flight op één echte
opname gedaan vóór deze run startte.
