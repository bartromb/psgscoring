# EDF+ analyseren met psgscoring — voor externe centra

*Wat u nodig hebt, wat u kiest, en waar het misgaat.*
*Hoort samen met [`aasm_cheatsheet.md`](aasm_cheatsheet.md).*

---

## 1. De kortste versie

1. Anonimiseer het EDF **vóór** verzending (§4).
2. Controleer of de montage de minimale kanalen bevat (§2).
3. Kies een profiel — `aasm_v3_rec` tenzij u weet waarom niet (§3).
4. Controleer in het rapport het **herkomstblok**: welk kanaal voedde welke
   analyse (§5). Klopt dat niet, dan klopt de rest ook niet.

## 2. Minimale montage

| Rol | Nodig voor | Zonder dit |
|---|---|---|
| EEG (bij voorkeur C4-M1 of C3-M2) | slaapstadiëring | geen TST → geen AHI |
| EOG | stadiëring (REM) | slechtere REM-detectie |
| **kin-EMG** | stadiëring (REM), arousals in REM | REM wordt onbetrouwbaar |
| nasale druk | hypopneeën | geen kwantitatieve flowbeoordeling |
| oronasale thermistor | apneus | apneus op de druk → overdetectie |
| thorax + abdomen (RIP) | obstructief/centraal onderscheid | geen typering |
| SpO₂ | desaturatiecriterium | Rule 1A/1B niet toepasbaar |
| pulse (los kanaal) | hartfrequentie | geen betrouwbare HR |
| PLMl / PLMr | PLM-analyse | geen PLM |

**Kin-EMG is geen been-EMG.** De REM-detectie steunt op kin-atonie; tibialis
anterior meet iets anders. Levert uw export beide, controleer welk kanaal de
software als staging-EMG gebruikt heeft — dat staat in het herkomstblok.

## 3. Kanaalnamen per fabrikant

De kanaalmapping bepaalt het resultaat, en de herkenning werkt op
**substring-matching** van de kanaalnaam. Enkele bekende gevallen:

| Fabrikant / dataset | Bijzonderheid |
|---|---|
| SOMNOmedics (DOMINO) | thermistor heet `Flow Th.` — te kort voor `"therm"`; beenkanalen heten `PLMl`/`PLMr` zonder scheidingsteken |
| NSRR (MESA, SHHS) | de nasale druk heet simpelweg `Pres` |
| PSG-IPA | geen thermistor in de montage; apneus vallen terug op de nasale druk |

Twee vallen die u zelf kunt controleren:

- **Geen los `Pulse`-kanaal?** Dan kan de hartfrequentierol een flowkanaal
  opeisen (`"pr"` is substring van `"Pres"`). Sinds v0.14.2 blokkeert
  `detect_channels()` dat, maar controleer het herkomstblok.
- **Meerdere EMG- of EOG-kanalen?** Dan bepaalt de volgorde in het bestand welk
  kanaal wordt voorgesteld. Kies handmatig als u het zeker wilt weten.

Wijkt uw montage af, geef dan een expliciete kanaalmap mee in plaats van op
auto-detectie te vertrouwen.

## 4. Anonimisatie

Verwijder uit de EDF-header: naam, geboortedatum, patiënt-ID en, indien
aanwezig, het opnamenummer van uw systeem. Vervang ze door een studienummer.

Let ook op:

- **EDF+-annotaties** kunnen identificatoren bevatten.
- De **bestandsnaam** draagt in de praktijk vaker een naam dan de header.
- De **opnamedatum** blijft nodig voor de analyse maar is in combinatie met een
  klein cohort herleidbaar; overleg wat uw ethische commissie hierover zegt.

## 5. Profielkeuze

| Profiel | Wanneer |
|---|---|
| `aasm_v3_rec` | **klinische standaard**: Rule 1A, 3 % of arousal |
| `cms_medicare` | wanneer een 4 %-criterium vereist is (Rule 1B) |
| `mesa_shhs` | reproductie van NSRR-analyses — niet voor klinisch gebruik |
| `aasm_v3_pressure` | zoals `aasm_v3_rec`, maar de vijf afgeleide analyses lezen de neusdruk |
| `aasm_v3_dual` | thermistor additief: apneus worden op beide sensoren gezocht en samengevoegd |

Vergelijk **nooit** getallen tussen profielen zonder het profiel te noemen. Het
verschil tussen 1A en 1B alleen al verschuift de AHI over ernstgrenzen heen.

## 6. Het herkomstblok lezen

Elk rapport bevat een tabel "Herkomst — welk kanaal voedde welke analyse":
staging-EEG/EOG/EMG, apneu- en hypopneukanaal, de flow-referentie van de
afgeleide analyses, de thermistorstatus, het profiel en beide softwareversies.

Dat blok is de reproduceerbaarheidsgarantie. Twee analyses van dezelfde nacht
zijn alleen vergelijkbaar als die tabel identiek is. Wijkt één regel af, dan is
het verschil in de uitkomst niet noodzakelijk klinisch.

## 7. Wat de software **niet** doet

- Hypoventilatie beoordelen (geen CO₂-analyse).
- Pediatrische criteria toepassen.
- Manuele scoring vervangen. Dit is een screening- en second-opinion-instrument;
  de eindbeoordeling blijft bij de arts.

## 8. Contact

Vragen over montage, profielkeuze of afwijkende resultaten: neem contact op vóór
u een reeks opnames analyseert. Eén proefopname met bekende manuele scoring
bespaart meer tijd dan honderd ongecontroleerde.
