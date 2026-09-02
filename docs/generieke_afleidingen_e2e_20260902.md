# Drie MESA-afleidingen tegen één: end-to-end

*Gemeten 2026-09-02, 90 opnames, disjunct van de afleidingsset.*

## De uitkomst

| arm | n events | index | bias | F1 |
|---|---:|---:|---:|---:|
| mens | 128 | 23,5 | — | — |
| EEG1 alleen (huidig) | 118 | 21,5 | −2,8 | 0,425 |
| alle drie | 154 | 27,6 | +1,4 | **0,512** |

Gepaard over 90 opnames: **ΔF1 mediaan +0,0706, beter op 85 van de 90,
p < 0,0001.** Dat brengt ons van **63 % naar 75 %** van het menselijke plafond
(F1 0,679, 330 scoorderparen).

Dit is de grootste arousalwinst die dit project heeft gemeten, en hij komt niet
uit een drempel maar uit twee kanalen die er altijd al lagen.

## De bias, in vier maten

Eén statistiek zou hier misleiden, dus alle vier:

| maat | één | drie |
|---|---:|---:|
| getekende bias | −2,83 | **+1,44** |
| \|bias\| per opname | 4,34 | 4,71 (p = 0,53) |
| RMS-bias | 11,05 | **9,80** |
| binnen ±5/u | 52 % | 51 % |

Drie van de vier zijn neutraal of beter. De vierde is nominaal slechter en
statistisch niet te onderscheiden van nul.

## Waarom de preregistratie tóch MISLUKT zegt

Criterium (b) luidde "de absolute indexbias verslechtert niet", en 4,71 > 4,34.

**Dat criterium was asymmetrisch gebouwd, en dat is een fout van mij:** voor de
WINST eiste ik p < 0,05, voor de SCHADE nam ik genoegen met een kale
medianenvergelijking. Een criterium in die vorm faalt ongeveer de helft van de
tijd op ruis alleen, ongeacht wat de wijziging doet.

De regel achteraf repareren op dezelfde data is precies waar preregistratie
tegen beschermt. De uitkomst blijft dus MISLUKT en de vlag blijft uit.

## Wat er in plaats daarvan gebeurt

Een **derde**, opnieuw disjuncte set van 90 opnames (38 per scoorder
overgeslagen), met de gecorrigeerde regel vooraf vastgelegd:

> slaagt als (a) mediane ΔF1 > 0 met p < 0,05, én (b) de absolute indexbias
> niet **aantoonbaar** verslechtert — dus niet: mediaan hoger *en* p < 0,05.

Symmetrisch: beide kanten vragen bewijs.

## De bredere les

`arousal_derivation_channels` koos op MESA één kanaal omdat `EEG1/EEG2/EEG3`
geen regiosleutel dragen. Elk MESA-arousalgetal dat dit project ooit
rapporteerde — het werkpunt 0,80, de artefactlijst, de alfaband, de
wake-epochs, F1 0,546 tegen het plafond — is dus op één afleiding gemeten,
terwijl de klinische doelmontage er drie heeft.

Dat maakt die metingen niet ongeldig, maar wel smaller dan ze leken: ze zeggen
iets over één-kanaals-MESA, niet over een klinische opname. Verschillende van
die knoppen verdienen een herhaling zodra deze vlag aanstaat.
