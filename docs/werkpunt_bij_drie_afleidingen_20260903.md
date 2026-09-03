# Drie afleidingen op werkpunt 0,80: geen ruil meer

*Gemeten 2026-09-03, 87 opnames (set 3), zes armen in één run.*

## De vraag

Drie afleidingen gaven op twee disjuncte sets een grote F1-winst, maar de
telling steeg op **elke** opname en dat kostte de index. Het vermoeden: het
LGBM-werkpunt van **0,70** is geijkt toen er één afleiding was, dus met drie
komt een ruimere kandidaatpool bij dezelfde drempel aan.

## De sweep

| arm | index | bias | \|bias\| | ≤5/u | F1 |
|---|---:|---:|---:|---:|---:|
| mens | 20,9 | — | — | — | — |
| **huidig** (één afleiding, 0,70) | 21,1 | −1,8 | 3,57 | 61 % | 0,438 |
| **drie op 0,80** | **20,8** | **−1,2** | **3,11** | 60 % | **0,556** |
| drie op 0,85 | 18,3 | −3,7 | 4,33 | 59 % | 0,569 |
| drie op 0,88 | 16,5 | −5,6 | 5,76 | 44 % | 0,567 |
| drie op 0,90 | 15,1 | −6,7 | 6,85 | 34 % | 0,570 |
| drie op 0,93 | 12,6 | −9,9 | 9,86 | 21 % | 0,549 |

*(drie afleidingen op de profieldrempel 0,70 gaf eerder index 25,2 / bias +3,3 /
F1 0,517 — de overtelling die deze sweep moest oplossen.)*

## Gepaard, 87 opnames

| arm | ΔF1 | beter op | p(F1) | Δ\|bias\| | p(bias) |
|---|---:|---:|---:|---:|---:|
| **drie 0,80** | **+0,0943** | **85/87** | 6,1e-16 | **−0,40** | **0,042** |
| drie 0,85 | +0,1139 | 86/87 | 5,7e-16 | +0,70 | 0,364 |
| drie 0,88 | +0,1275 | 85/87 | 6,1e-16 | +2,30 | 0,0034 |
| drie 0,90 | +0,1337 | 80/87 | 1,9e-15 | +3,40 | <1e-4 |
| drie 0,93 | +0,1164 | 74/87 | 4,8e-11 | +5,60 | <1e-4 |

**Op 0,80 is er geen ruil.** De F1 stijgt van 0,438 naar 0,556 — van **65 % naar
82 %** van het menselijke plafond (0,679) — en de absolute indexbias wordt
*significant beter*, niet slechter. De index komt op 20,8 tegen een menselijke
20,9.

Boven 0,80 koopt elke stap nog wat F1 maar breekt de index af: op 0,90 zit nog
maar 34 % van de opnames binnen ±5/u, tegen 61 % nu.

## Wat hier nog niet aan deugt

De drempel 0,80 is **gekozen op deze set**. Dat is een fit. De afleidingsvlag
zelf repliceerde al op twee disjuncte sets, maar de páárvorming met 0,80 niet.

Er draait daarom een bevestiging op een **vijfde** disjuncte set met de
gecorrigeerde symmetrische regel vooraf vastgelegd: huidige productie (één
afleiding, 0,70) tegen drie afleidingen op 0,80.

Slaagt die, dan is dit de eerste arousalwijziging van dit project die zowel de
localisatie als de index verbetert — en de grootste sprong tot nu toe.

## Terzijde, over het werkpunt

Commit 9d79dc0 verplaatste het werkpunt op 2026-08-30 bewust van 0,80 naar 0,70,
samen met de 10 s-intervalregel. Met drie afleidingen komt 0,80 dus terug als
de juiste waarde — niet als terugdraaiing, maar omdat de pool waarop de drempel
werkt is veranderd.


---

# Bevestigd op de vijfde set: GESLAAGD op elke maat

*Toegevoegd 2026-09-03, 89 opnames, vijfde disjuncte set, regel vooraf vastgelegd.*

| arm | index | bias | \|bias\| | ≤5/u | F1 |
|---|---:|---:|---:|---:|---:|
| mens | 22,2 | — | — | — | — |
| huidige productie (één afleiding, 0,70) | 20,9 | −3,5 | 5,75 | 46 % | 0,441 |
| **drie afleidingen, 0,80** | 21,6 | **−2,6** | **4,37** | **57 %** | **0,555** |

Gepaard: **ΔF1 +0,1079, beter op 89 van de 89**, p = 2,6e-16.
**\|bias\| 5,75 → 4,37, p < 0,0001** — significant beter.
Binnen ±5/u: **46 % → 57 %**.

**PREREGISTRATIE: GESLAAGD.**

## Het totaalbeeld over vijf disjuncte sets

| set | n | vergelijking | ΔF1 | beter op |
|---|---:|---|---:|---:|
| 2 | 90 | drie tegen één, drempel 0,70 | +0,0706 | 85/90 |
| 3 | 87 | drie tegen één, drempel 0,70 | +0,0666 | 83/87 |
| 3 | 87 | drie op 0,80 tegen productie | +0,0943 | 85/87 |
| **5** | **89** | **drie op 0,80 tegen productie** | **+0,1079** | **89/89** |

Beter op élke opname van de vijfde set. De F1 gaat van 65 % naar 82 % van het
menselijke plafond, en anders dan bij alle eerdere arousalknoppen verbetert de
index mee in plaats van eronder te lijden.

## Wat dit voor uitrol betekent

De twee wijzigingen zijn **gekoppeld**: `arousal_generic_derivations = True`
zonder de drempel op 0,80 geeft juist overtelling (index 25,2 tegen 20,9).
Ze horen samen aan of samen uit.

## De vraag die hier direct uit volgt

Op een KLINISCHE montage (F4-M1, C4-M1, O2-M1) kiest de picker **nu al** drie
afleidingen — daar verandert de vlag niets. Maar die opnames draaien wél op
drempel 0,70, en dit werk laat zien dat 0,80 bij drie afleidingen beter is.

Als dat ook op PSG-IPA geldt, dan draait élke klinische opname nu op een
werkpunt dat voor één afleiding was geijkt. Dat is een grotere zaak dan de vlag
zelf, en het is meteen te meten: PSG-IPA heeft drie echte afleidingen en twaalf
scoorders.
