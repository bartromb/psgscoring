# Wat de +2 s onsetverschuiving met AHI en RDI doet — MESA n=30, gepaard

Gemeten 2026-08-26. Beantwoordt de blokkerende vraag uit
`arousal_onsetverschuiving_mesa_bevinding.md`: arousals zijn niet alleen
uitkomst maar ook **invoer** — ze bevestigen Rule 1B-hypopneus (venster 15 s) en
dragen de RERA-detectie. Een verschuiving van 2 s kan de klinische index
veranderen ook al blijft de arousaltelling identiek.

## Opzet

Dezelfde 30 opnames (`--seed 20260801`), dezelfde pipeline, dezelfde profielen;
het enige verschil is dat elke gedetecteerde arousal 2 s later ligt. De
verschuiving grijpt aan **vóór** de koppeling en de RERA-stap, want daarna
schuiven verandert alleen wat er gerapporteerd wordt, niet wat de index voedt.

Beide armen draaien door dezelfde injectie (`sitecustomize`, offset 0 of 2), zodat
alleen het getal verschilt. Bewijs dat de injectie in de workers aankwam: 14
respectievelijk 8 markeringen, met `offset=0.0` en `offset=2.0`.

## Uitkomst — `aasm_v3_breath` (draagt hypopneu-arousal én RERA/RDI)

| | 0 s | +2 s | gepaarde Δ | anders op | ernstklasse |
|---|---:|---:|---:|---:|---:|
| **AHI** (mediaan) | 14,40 | 14,40 | +0,01 gem | 10/30 (p = 1) | **0/30** |
| **RDI** (mediaan) | 21,75 | 21,95 | +0,09 gem | 26/30 (p = 0,56) | **0/30** |
| AHI-bias vs `aasm15` | −6,72 | −6,77 | | | |
| resp. event-F1 vs `aasm15` | 0,4918 | 0,4920 | −0,0011 (p = 0,75) | | |

**Controle gehaald:** de arousaltelling is identiek op **30/30**. Een
verschuiving verplaatst events en maakt er geen; was dat niet zo, dan mat dit
iets anders dan bedoeld.

## Uitkomst — `aasm_v3_rec` (controleprofiel)

Niets verandert: AHI, RDI en event-F1 zijn identiek op **0/30 afwijkingen**.
Dat bevestigt wat bij de classifier-uitrol al bleek — dit profiel reageert niet
op de arousal-lijst — en het laat zien dat de meting geen ruis oppikt.

## Lezing

**De verschuiving is klinisch gratis.** De arousal-localisatie verbetert
(+0,0140 F1 op MESA, +0,0123 op PSG-IPA, beide vooraf vastgelegd) zonder dat de
AHI, de RDI, de ernstklasse of de respiratoire event-F1 er noemenswaardig op
reageert.

Twee nuances die niet weggepoetst moeten worden:

1. **De RDI reageert wél**, op 26 van de 30 opnames — alleen heel klein
   (mediaan +0,05/u) en zonder één ernstklasseverschuiving. Dat is te
   verwachten: de RERA-detectie is gevoeliger voor arousal-timing dan de
   Rule 1B-bevestiging, die een venster van 15 s heeft waar 2 s zelden
   doorheen breekt. De AHI verandert dan ook op maar 10 van de 30.
2. **Gratis is niet hetzelfde als goed.** Deze meting zegt dat de prijs nul is,
   niet dat de winst groot is. De winst is +0,014 arousal-F1.

Ter vergelijking: de arousal-classifier op de RERA-dragende profielen (23-08)
kostte een **RDI-ernstklasseverschuiving op 11/30 = 37 %**. Dat is de maat
waarmee deze nul moet worden gelezen.

## Wat hierna nodig is

De voorwaarde uit de voorregistratie is hiermee vervuld. Een profielvlag
(`arousal_onset_offset_s`, default 0,0) met tests is nu gerechtvaardigd. Een
default-flip blijft een aparte beslissing: de gerapporteerde arousal-onsets in
het klinische rapport schuiven mee, ook al verandert de index niet.

## Reproductie

`scripts/onset_offset_ab/run_ab.sh` + `scripts/onset_offset_ab/sitecustomize.py`,
`validate_mesa.py
--n 30 --seed 20260801 --profiles aasm_v3_breath`; vergelijking met
`scripts/compare_arousal_onset_offset_ahi.py`. Ruwe uitvoer: `mesa_off0.json`,
`mesa_off2.json` (niet in de repo; de JSON-uitvoer staat naast de meting).
