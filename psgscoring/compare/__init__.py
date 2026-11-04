"""Interoperabiliteit met externe scoringssystemen, voor vergelijking.

Alleen LEZERS: elke module hier vertaalt de uitvoer van een ander systeem
naar het eventschema van psgscoring. Geen van deze modules importeert of
levert code van dat systeem mee, en geen ervan zit in het scoringspad — een
import hieruit mag nooit een gescoorde waarde kunnen bewegen.
"""

from .caisr_reader import (  # noqa: F401
    CAISR_RESP_CODES,
    ahi_from_events,
    labels_to_events,
    read_caisr_resp_csv,
    verify_code_mapping,
)
from .domino_reader import (  # noqa: F401
    DOMINO_LIMB_LABELS,
    DOMINO_RESP_LABELS,
    DOMINO_STAGE_LABELS,
    DominoExport,
    DominoRow,
    DominoSignal,
    NotDominoFile,
    ahi_from_domino_events,
    estimate_clock_offset,
    hypnogram_kappa,
    parse_domino_sheet,
    parse_domino_txt,
    read_domino_export,
)

__all__ = [
    "CAISR_RESP_CODES",
    "ahi_from_events",
    "labels_to_events",
    "read_caisr_resp_csv",
    "verify_code_mapping",
    "DOMINO_LIMB_LABELS",
    "DOMINO_RESP_LABELS",
    "DOMINO_STAGE_LABELS",
    "DominoExport",
    "DominoRow",
    "DominoSignal",
    "NotDominoFile",
    "ahi_from_domino_events",
    "estimate_clock_offset",
    "hypnogram_kappa",
    "parse_domino_sheet",
    "parse_domino_txt",
    "read_domino_export",
]
