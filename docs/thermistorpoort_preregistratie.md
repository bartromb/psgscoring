# Preregistratie — welke poort beslist of apneus op de thermistor gescoord worden

*Opgesteld 22-08-2026, vóór enige uitkomst op MESA. De verkenning van de
doorlaatpercentages liep nog toen dit werd vastgelegd.*

## Wat er nu gebeurt

`assess_flow_sensor_agreement` bepaalt op **elke** opname of apneus op de
thermistor of op de neusdruk gescoord worden. Negentien van de eenentwintig
profielen gebruiken hem, waaronder `aasm_v3_rec`, `aasm_v3_breath` en
`aasm_v3_prob`. Alleen `aasm_v3_breath_dual` en `aasm_v3_prob_dual` staan op
`respiratory_band`.

De maat is op 06-08-2026 weerlegd als maat voor de vraag die hij beantwoordt.
Zes synthetische signalen die **allemaal op 0,25 Hz ademen** geven agreement
1,000 tot −0,985, afhankelijk van hun trage amplitudemodulatie:

| geval | agreement |
|---|---:|
| identieke amplitudemodulatie | 1,000 door |
| thermistor 1 s vertraagd | 1,000 door |
| modulatie 90° verschoven | 0,002 af |
| geen modulatie op de thermistor | 0,094 af |
| modulatie in tegenfase | −0,985 af |

De maat reageert uitsluitend op overeenstemming in trage amplitudemodulatie
en is blind voor de vraag of beide sensoren dezelfde ademhaling zien. Op negen
lokale montages haalde **één** de drempel, terwijl vier spectraal gemeten
opnames exact dezelfde ademfrequentie op beide sensoren lieten zien en de
thermistor het schónere ademspectrum had.

Dat maakt de poort verdacht, **niet aantoonbaar schadelijk**: een poort kan
op de verkeerde grond oordelen en tóch de betere sensorkeuze maken. Dat is wat
hier gemeten wordt.

## De vraag

Scoort `respiratory_band` — een toets op één kanaal, adembandvermogen ≥ 0,70 —
dichter bij menselijke scoring dan `envelope_agreement`?

## Opzet

- **Cohort:** MESA, gepaarde run over dezelfde opnames, seed vastgelegd in het
  runscript. MESA heeft `Pres` én `Therm`, dus de poort draait echt.
- **Armen:** `PSGSCORING_THERMISTOR_GATE=envelope_agreement` tegen
  `respiratory_band`. Env-override, zodat de registry niet muteert en beide
  armen aantoonbaar op hetzelfde profiel draaien.
- **Profiel:** `aasm_v3_breath`.
- **Referentie:** de MESA-annotaties (apneus en hypopneus).
- **Maten:** event-F1 met greedy IoU-koppeling op 0,20, en AHI-bias.

## Beslisregel — vastgelegd vóór de meting

`respiratory_band` wordt default op de niet-gepinde v3-profielen **alleen
als beide** waar zijn:

1. mediane gepaarde ΔF1 ≥ **+0,010**, en
2. de mediane gepaarde AHI-bias verslechtert met niet meer dan **1,0/u** in
   absolute waarde.

**Weerlegd** als de mediane gepaarde ΔF1 ≤ 0. Daartussen: blijft opt-in, en
ik rapporteer het als onbeslist in plaats van het alsnog aan te zetten.

Gepind op `envelope_agreement`, ongeacht de uitkomst: `mesa_shhs`,
`chicago_1999`, `cms_medicare`, `aasm_v1_rec`, `aasm_v2_rec`. Die reproduceren
een externe regelset of paper v31/v37.

## Meldgrens

Verschuift de **AHI-ernstklasse** op meer dan een kwart van de opnames, dan
leg ik dat als apart punt voor in plaats van het in een verslag te vermelden.
Dezelfde grens als bij de RDI-karakterisering, en om dezelfde reden: een
sensorwissel die een kwart van de patiënten herklasseert is een beslissing,
geen detail.

## Wat deze meting NIET kan uitwijzen

Of de poort op de júiste grond oordeelt. Beide poorten worden hier op hun
uitkomst afgerekend, niet op hun redenering. Een poort die om de verkeerde
reden de goede sensor kiest, wint hier — en dat is voor een klinische keuze
verdedigbaar, maar het betekent dat een winst hier de synthetische weerlegging
van 06-08-2026 niet ongedaan maakt.

Evenmin uit te wijzen: hoe dit op de lokale AZORG-montages uitpakt. Daar is
geen menselijke referentie. De negen montages tonen alleen dat de poorten
verschillend oordelen, niet wie gelijk heeft.

## Derde poort

`breath_coherence` bestaat ook en wordt door geen enkel profiel gebruikt. Die
blijft hier buiten beschouwing: twee armen tegelijk vergelijken met één
beslisregel maakt de regel dubbelzinnig. Het doorlaatpercentage wordt wel
gerapporteerd.
