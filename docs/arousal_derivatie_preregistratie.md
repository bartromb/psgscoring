# Preregistratie — single versus multi-derivatie

Datum: 2026-08-22. **Geschreven vóór de meting.**

## Waarom dit geen verfijning is maar een gat

`AROUSAL_DERIVATION_MODE` staat op **multi** op 23 van de 24 profielen; alleen
`mesa_shhs` draait single. Multi is dus wat er in productie draait.

**Alle arousalmetingen van 21–22 augustus draaiden op single-derivatie**, op
één centraal kanaal, omdat het harnas `detect_arousals` rechtstreeks aanroept.
Het cijfer waarop de hybride default is gezet — event-F1 0,463 tegen 0,692
menselijk — geldt dus voor een configuratie die niet de default is.

Datzelfde geldt voor de MESA n=150-meting: daar vindt `_pick_eeg_multi` maar
één afleiding (de kanalen heten `EEG1/2/3` en matchen de patronen voor
occipitaal en frontaal niet), zodat multi vanzelf naar single degradeert. Ook
die AHI-cijfers zijn dus single-derivatie.

Er is bovendien een eerdere meting op het REGELgebaseerde pad die tegen multi
pleit: die gaf hogere sensitiviteit (0,38 → 0,47) bij een vlakke F1 (~0,38) —
winst in dekking, niet in accuraatheid. Of dat met een classifier erachter
anders ligt, is nooit gemeten.

## De ingreep

Drie armen op de vijf PSG-IPA arousal-opnames, alle met de hybride aan
(drempel 0,60), tegen dezelfde twaalf scoorders:

- **single** — één centraal kanaal (wat ik tot nu toe mat)
- **multi** — centraal + occipitaal + frontaal, event-level union
- **multi + EOG-reject** — idem, met verwerping van occipitaal-only events die
  samenvallen met een oogbeweging

PSG-IPA draagt `EEG F4-M1`, `EEG C4-M1` (of `Cz-M1`), `EEG O2-M1` en
`EOG E1-M2`/`E2-M2`, dus alle drie zijn meetbaar.

## Acceptatiecriterium (vastgelegd vóór de meting)

**Primair.** De mediane event-F1 van **multi** ligt boven die van single
(0,463 in dezelfde run gemeten, niet uit een eerdere overgenomen). Haalt multi
dat niet, dan is de huidige default op 23 profielen niet gerechtvaardigd en
hoort de vraag omgekeerd te worden gesteld: waarom staat multi aan?

**Secundair (mechanisme).** De recall stijgt met multi. Meer afleidingen horen
meer van wat de scoorders markeerden te vinden; gebeurt dat niet, dan doet de
union iets anders dan verondersteld.

**Bewaking.** De spreiding van `index_algoritme / index_scoordermediaan` blijft
binnen de 0,61–1,29 van single.

**Wat hier NIET uit volgt.** Ook als multi wint, verandert er geen default
zonder een meting van de respiratoire gevolgen — arousals voeden Rule 1B en de
RDI. En omgekeerd: verliest multi, dan is dat een argument om de bestaande
default te herzien, wat óók een gebruikersbeslissing is.

## Verwachting

Op grond van de eerdere meting op het regelpad: recall omhoog, F1 vlak of
lager. Als dat zich herhaalt mét classifier, is de conclusie dat de union
dekking koopt met precisie — en dat de huidige default die afruil maakt zonder
dat iemand hem gemeten heeft.
