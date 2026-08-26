"""Verschuift arousal-onsets met een vast aantal seconden, vóór de koppeling.

WAAROM sitecustomize EN NIET EEN MONKEYPATCH IN HET HARNAS
Python importeert dit in ELK proces bij het opstarten, dus ook in de workers van
`ProcessPoolExecutor` -- ongeacht of die geforkt of gespawnd worden. Een patch
die alleen in het ouderproces staat, is bij `spawn` stilzwijgend weg en levert
een A/B waarin beide armen identiek zijn.

WAAROM HIER EN NIET NA `run_arousal_respiratory_analysis`
Die functie koppelt de arousals BINNEN zichzelf aan de respiratoire events en
draait er de RERA-detectie op. Na afloop schuiven verandert alleen wat er
gerapporteerd wordt, niet wat de AHI en de RDI voedt -- en juist dat is de vraag.

DUBBELE VERSCHUIVING VERMIJDEN
`detect_arousals_multi` roept intern `detect_arousals` aan. Zonder guard zou de
binnenste patch schuiven en de buitenste er nog eens overheen: +4 in plaats van
+2. De vlag `_binnen_multi` voorkomt dat.
"""
import os

_OFF = float(os.environ.get("PSGSCORING_AROUSAL_ONSET_OFFSET_S", "0") or 0)
_MARK = os.environ.get("PSGSCORING_AROUSAL_ONSET_MARKFILE")

if _MARK is not None:
    try:
        import psgscoring.arousal as _A
    except Exception:                                    # noqa: BLE001
        _A = None

    if _A is not None:
        _multi_orig = _A.detect_arousals_multi
        _single_orig = _A.detect_arousals
        _binnen_multi = False
        _gemeld = False

        def _meld(fn):
            global _gemeld
            if _gemeld:
                return
            _gemeld = True
            try:
                with open(_MARK, "a") as f:
                    f.write(f"{os.getpid()} {fn} offset={_OFF}\n")
            except OSError:
                pass

        def _schuif(res):
            if _OFF and res and res.get("events"):
                for e in res["events"]:
                    for k in ("onset_s", "end_s"):
                        if e.get(k) is not None:
                            e[k] = float(e[k]) + _OFF
            return res

        def _multi(*a, **k):
            global _binnen_multi
            _meld("multi")
            _binnen_multi = True
            try:
                res = _multi_orig(*a, **k)
            finally:
                _binnen_multi = False
            return _schuif(res)

        def _single(*a, **k):
            _meld("single")
            res = _single_orig(*a, **k)
            return res if _binnen_multi else _schuif(res)

        _A.detect_arousals_multi = _multi
        _A.detect_arousals = _single
