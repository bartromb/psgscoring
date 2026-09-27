# data/test — testopnames voor bench/

Deze map is **gitignored** (op dit bestand na). Ze bevat uitsluitend:

- publieke datasets met een licentie die lokaal gebruik toestaat
  (bv. PSG-IPA van PhysioNet), of
- synthetische opnames (`myproject/generate_demo_edf.py` van
  YASAFlaskified, of eigen generatoren onder `bench/`).

**Nooit patiëntdata**, ook niet tijdelijk, ook niet geanonimiseerd.
NSRR-data (MESA/SHHS) valt onder een DUA: gebruik die vanaf `/srv/DATA`
via een pad in het benchmarkscript, kopieer ze niet hierheen.

De echte benchmarkdata staat op `/srv/DATA` (PSG-IPA publiek; MESA en SHHS
onder DUA — ter plekke gebruiken, niets kopiëren, niets schrijven behalve de
registerregel in `MESA/gebruikte_mesa_ids.txt`). Referenties exporteer je
met `bench/export_ref.py` naar het formaat dat `bench/evaluate.py` leest:
`<naam>_ref.csv` met kolommen `onset_s,offset_s[,type]` (of `_ref.json`).
Hier horen alleen synthetische of kleine fixtures.
