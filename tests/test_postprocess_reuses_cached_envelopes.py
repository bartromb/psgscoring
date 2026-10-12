"""De gemengde-apneudecompositie krijgt de effort-enveloppen uit de
voorbewerkingscache van de pipeline en rekent ze niet opnieuw (profilering
12-10-2026: +17 % looptijd op `aasm_v3_breath_dual` door dubbele
`preprocess_effort`-passes). Byte-identiek: dezelfde functie, dezelfde invoer."""
from __future__ import annotations

from psgscoring import pipeline, signal as psignal
from tests.test_golden_output import CASES, make_raw


def test_decompositie_roept_preprocess_effort_niet_zelf_aan(monkeypatch):
    cfg = dict(CASES["apnea_clean"]); cfg.pop("profile"); cfg.pop("arousals", None)
    raw, hypno, channel_map = make_raw(**cfg)
    telling = {"n": 0}
    # postprocess importeert `preprocess_effort` lokaal uit psgscoring.signal;
    # respiratory heeft zijn eigen gebonden naam en telt hier niet mee.
    origineel = psignal.preprocess_effort

    def _spy(*args, **kwargs):
        telling["n"] += 1
        return origineel(*args, **kwargs)

    monkeypatch.setattr(psignal, "preprocess_effort", _spy)
    out = pipeline.run_pneumo_analysis(raw, hypno, channel_map=channel_map,
                                       scoring_profile="aasm_v3_breath_dual")
    assert out["respiratory"]["success"]
    assert telling["n"] == 0, f"postprocess rekende de enveloppen {telling['n']}x opnieuw"
