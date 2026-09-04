# Externe benchmark: psgscoring naast CAISR/ABED op 50 MESA-opnames

*2026-09-04. Eerste vergelijking op gelijke voet: zelfde opnames, zelfde
NSRR-staging als noemer voor beide, zelfde matcher (IoU 0,20), beide tegen de
menselijke NSRR-events. CAISR-App lokaal gedraaid (CC BY-NC 4.0);
`caisr_resp` is van de eerste ABED-auteur en is functioneel het
gepubliceerde detectorpad.*

## Uitkomst

| | mens | psgscoring 0.32.0 | CAISR/ABED |
|---|---:|---:|---:|
| AHI (mediaan) | 25,5 | **25,0** | 69,3 |
| \|AHI-bias\| (mediaan) | — | **7,1** | 40,3 |
| events per opname | 153 | **146** | 420 |
| event-F1 tegen NSRR | — | **0,410** | 0,234 |
| Spearman AHI met mens | — | 0,578 | 0,512 |

Gepaard op F1: **Δ +0,178, psgscoring beter op 48/50, Wilcoxon p = 1,9e-13.**

## Kanttekeningen die net zo hard gelden als de winst

1. **CAISR draaide buiten zijn thuisbasis.** MESA's generieke kanaalnamen
   (`EEG1/2/3`) dwongen hun arousalmodule in de 2-kanaals-terugval — dezelfde
   val die onze eigen picker deze week bleek te hebben — en hun
   3 %-óf-arousal-hypopneutak valideert tegen die eigen arousalstroom. De
   overtelling (3× de mens) past bij een oversensitieve arousalinvoer. Dit is
   dus "CAISR zoals nu inzetbaar op MESA", niet "ABED op zijn eigen cohorten"
   (papier: apneu-F1 0,78 met hún harnas en matching — niet vergelijkbaar).
2. **Hun 1 Hz-labelvector** voegt aangrenzende events van hetzelfde type
   samen (mediaan 8/opname) — drukt hun F1 beperkt, verklaart de telling niet.
3. **Twee lokale compat-fixes** waren nodig om hun keten op MESA te laten
   lopen (read-only mV-pad in de EDF-loader; pandas-view in de chin-route),
   beide gemarkeerd in hun bomen; en per opname is hún staging vervangen door
   het NSRR-hypnogram zodat de noemers identiek zijn.

## Wat dit wel en niet zegt

* Onze 0,410 tegen de NSRR-enkelscoorder is consistent met het interne beeld
  (menselijk plafond op multi-scoorderdata 0,556; één referentiescoorder
  drukt elke F1).
* De claim is smal en precies: **op MESA-klasse montages levert psgscoring een
  menselijk-correcte telling waar het sterkste beschikbare externe systeem
  drievoudig overtelt, en betere localisatie op 48 van de 50 nachten.**
* Voor de paper is dit de eerste externe referentie op gelijke voet; de
  CAISR-arousal-CSV's liggen er en maken dezelfde vergelijking voor arousals
  mogelijk.

Bestanden: `/home/bart/caisr_mesa/` (ids, CSV's, JSONs,
`caisr_vs_psgscoring_mesa.csv`).
