"""Het harnas mag niet de hele nacht als artefact vlaggen.

`_artifact_epochs_like_production` selecteerde EEG-kanalen met
`startswith("EEG")` en pakte daarmee op MESA ook `EEG1_Off`..`EEG3_Off` mee --
offsetkanalen met pieken van MILJOENEN uV. Elk epoch haalde de 500 uV-drempel,
dus 100 % van de nacht gold als artefact en de gemeten AHI kwam op nul uit.
Ik las dat eerst als een effect van artefactonderdrukking.

Productie heeft dat probleem niet: die krijgt `all_eeg_channels` uit de
kanaaldetectie. Alleen deze replicatie had het.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _raw(extra=None):
    mne = pytest.importorskip("mne")
    sf, n_ep = 64.0, 40
    n = int(sf * 30 * n_ep)
    rng = np.random.default_rng(2)
    kanalen = {"EEG1": rng.normal(0, 20e-6, n), "EEG2": rng.normal(0, 20e-6, n)}
    if extra:
        kanalen.update(extra(n, rng))
    info = mne.create_info(list(kanalen), sf, ["eeg"] * len(kanalen))
    return mne.io.RawArray(np.vstack(list(kanalen.values())), info, verbose=False)


def test_offset_channels_do_not_flag_the_whole_night():
    pytest.importorskip("mne")
    from validate_mesa import _artifact_epochs_like_production

    def offsets(n, rng):
        return {"EEG1_Off": rng.normal(0, 3.0, n),      # volt-schaal = miljoenen uV
                "EEG2_Off": rng.normal(0, 3.0, n)}

    raw = _raw(offsets)
    art = _artifact_epochs_like_production(raw)
    n_ep = 40
    assert len(art) < n_ep, (
        f"{len(art)}/{n_ep} epochs gevlagd -- de offsetkanalen worden nog "
        "meegenomen en vlaggen de hele nacht")
    assert len(art) == 0, (
        f"schoon EEG hoort niets te vlaggen, kreeg {sorted(art)[:5]}")


def test_a_real_artefact_is_still_flagged():
    """Zonder deze controle zou 'nul gevlagd' ook slagen op een kapotte functie."""
    pytest.importorskip("mne")
    from validate_mesa import _artifact_epochs_like_production

    raw = _raw()
    dat = raw.get_data()
    sf = raw.info["sfreq"]
    epl = int(30 * sf)
    for ep in (7, 19):
        dat[:, ep * epl:(ep + 1) * epl] *= 60      # ver boven 500 uV
    raw._data = dat

    art = _artifact_epochs_like_production(raw)
    assert {7, 19} <= set(art), f"echt artefact gemist: {sorted(art)}"
    assert len(art) < 40, "vlagt alsnog de hele nacht"
