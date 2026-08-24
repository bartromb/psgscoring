"""Het gebundelde model, geïnspecteerd — de grond onder de EMG-guard.

De guard in ``detect_arousals`` (geen bruikbaar kin-EMG → regelgebaseerd pad)
rust op twee eigenschappen van ``arousal_classifier_v3.txt``. Beide worden
hier gemeten in plaats van aangenomen, zodat een toekomstig v4-model dat de
aanname niet meer waarmaakt de guard MEE laat verhuizen in plaats van hem
stilzwijgend overbodig of schadelijk te maken.

1. Het model splitst zwaar op ``emg_var_ratio`` en alle drempels liggen boven
   nul. Zonder EMG staat dat feature constant op 0,0, dus gaat elke kandidaat
   in elke split dezelfde kant op en schuift de hele kansverdeling — op een
   vast werkpunt is dat een decimering, geen graduele versoepeling.

2. Het model splitst NERGENS op ``emg_confirmed``. Dat is de vergunning voor
   de tweede wijziging (``emg_confirmed`` is niet langer True als default):
   de voorspellingen kunnen er per constructie niet door bewegen.
"""
import pytest

from psgscoring.arousal import (
    _AROUSAL_LGBM_FEATURE_ORDER,
    AROUSAL_LGBM_MODEL_PATH,
)


@pytest.fixture(scope="module")
def splits():
    lgb = pytest.importorskip("lightgbm")
    import os
    if not os.path.exists(AROUSAL_LGBM_MODEL_PATH):
        pytest.skip("model niet gebundeld in deze installatie")
    dump = lgb.Booster(model_file=AROUSAL_LGBM_MODEL_PATH).dump_model()
    per_feature: dict[int, list[float]] = {}
    trees: dict[int, set] = {}

    def walk(node, ti):
        if "split_feature" not in node:
            return
        f = node["split_feature"]
        per_feature.setdefault(f, []).append(node["threshold"])
        trees.setdefault(f, set()).add(ti)
        walk(node["left_child"], ti)
        walk(node["right_child"], ti)

    for ti, tree in enumerate(dump["tree_info"]):
        walk(tree["tree_structure"], ti)
    return per_feature, trees, len(dump["tree_info"])


def test_the_model_leans_on_emg_var_ratio(splits):
    per_feature, trees, n_trees = splits
    i = _AROUSAL_LGBM_FEATURE_ORDER.index("emg_var_ratio")
    ths = per_feature.get(i, [])
    assert len(ths) > 100, (
        f"nog maar {len(ths)} splits op emg_var_ratio — als het model hier "
        f"niet meer op leunt, hoort de EMG-guard opnieuw gemeten te worden"
    )
    assert len(trees.get(i, ())) > n_trees // 4
    assert min(ths) > 0.0, (
        "een drempel op of onder nul zou betekenen dat 'geen EMG' (0,0) ook "
        "eens de andere kant op gaat; de degeneratie-redenering klopt dan niet"
    )


def test_the_model_never_splits_on_emg_confirmed(splits):
    """Zonder deze eigenschap zou het herstellen van `emg_confirmed` de
    voorspellingen verschuiven en was het geen kosteloze reparatie."""
    per_feature, _trees, _n = splits
    i = _AROUSAL_LGBM_FEATURE_ORDER.index("emg_confirmed")
    assert per_feature.get(i, []) == [], (
        f"het model splitst {len(per_feature.get(i, []))} keer op "
        f"emg_confirmed — de semantiekwijziging is dan niet gedragsneutraal"
    )


def test_the_feature_order_still_matches_the_model(splits):
    """De kolomvolgorde is positioneel: het model draagt generieke namen
    (Column_0..Column_49). Een verschoven lijst maakt elke inspectie hierboven
    én elke voorspelling betekenisloos, zonder ooit een fout te geven."""
    per_feature, _trees, _n = splits
    assert len(_AROUSAL_LGBM_FEATURE_ORDER) == 50
    assert max(per_feature) < len(_AROUSAL_LGBM_FEATURE_ORDER)
