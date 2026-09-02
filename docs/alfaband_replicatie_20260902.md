# Alfaband 8–13 Hz: de F1-winst repliceert, de biaswinst niet

*Gemeten 2026-09-02. Vooraf vastgelegde beslisregel, uitkomst: MISLUKT.*

## De aanleiding

Op 24 MESA-opnames was de brede alfaband de enige arousalwijziging van
2026-09-01 die op **beide** maten tegelijk de goede kant op ging:

| arm | index | bias | F1 |
|---|---:|---:|---:|
| 8–11 Hz (huidig) | 22,6 | −5,8 | 0,443 |
| 8–13 Hz | 24,1 | −4,7 | 0,453 |

Dat was opmerkelijk: bij het werkpunt, de wake-epochs en de harde poort bewogen
telling en localisatie steeds tegen elkaar in.

## De preregistratie

Vastgelegd vóór de uitkomst, in `repliceer_alfa.py`:

* **set** — 90 opnames, 30 per MESA-scoorder, **disjunct** van de 24 waarop het
  effect is afgeleid (overlap geverifieerd op 0);
* **maat** — gepaard per opname, dezelfde opname in beide armen;
* **slaagt** als (a) mediane ΔF1 > 0 met Wilcoxon p < 0,05, **én** (b) de
  absolute indexbias niet verslechtert.

## De uitkomst

| arm | n events | index | bias | F1 |
|---|---:|---:|---:|---:|
| mens | 128 | 23,5 | — | — |
| 8–11 Hz | 118 | 21,5 | −2,8 | 0,425 |
| 8–13 Hz | 118 | 22,5 | −2,0 | 0,438 |

Gepaard over 90 opnames:

| criterium | uitkomst | oordeel |
|---|---|---|
| ΔF1 (8–13 min 8–11) | mediaan **+0,0049**, beter op 56/90, **p = 0,024** | gehaald |
| \|bias\| | 4,34 → **4,39**, p = 0,046 | **niet gehaald** |

**De regel is niet gehaald: de vlag gaat niet aan.**

## Wat er precies gebeurde

De F1-helft repliceert. De biashelft niet — die draait om. Op de afleidingsset
ging de bias van −5,8 naar −4,7 (beter); op de replicatieset van −2,8 naar −2,0
in de ruwe waarde, maar de **absolute** bias per opname wordt gemiddeld iets
groter, want de brede band tilt óók opnames op die al te hoog telden.

Dit is hetzelfde patroon als bij `rectify_lowpass`: beter op PSG-IPA, teken
omgedraaid op MESA. De regel deed precies waarvoor hij bestaat.

## Wat hier tegenover staat

Er is een conformiteitsargument los van de prestatie. De alfaband heet in de
literatuur en in de meeste scoringssoftware 8–13 Hz; onze 8–11 Hz is een
nauwere keuze die nergens is verantwoord. De AASM-arousalregel zelf noemt geen
band — ze spreekt van "alpha, theta and/or frequencies greater than 16 Hz" —
dus de manual dwingt niets af.

Het verschil is klein in beide richtingen: F1 0,425 tegen 0,438, index 21,5
tegen 22,5, tegen een menselijke 23,5. Wie op conformiteit wil sturen kan
verdedigen dat 8–13 hoort; wie op de vooraf vastgelegde regel stuurt houdt
8–11. Dat is een keuze, geen meetuitkomst, en hij ligt bij de gebruiker.

**Aanbeveling: vlag uit laten.** De regel bestaat juist om niet achteraf te
onderhandelen nadat de cijfers binnen zijn.

## Nevenvondst

De bias verschilt sterk tussen de twee sets: −5,8 op de afleidingsset van 24,
−2,8 op de disjuncte 90. Zelfde detector, zelfde profiel, zelfde scoorders.
Dat is een waarschuwing voor elk arousalgetal uit een kleine MESA-steekproef.
