"""Data provenance taxonomy and eligibility enforcement."""

from enum import Enum
from typing import Set


class DataProvenance(str, Enum):
    """Rigorous classification of data origin and authenticity.

    REAL_HUMAN:
        Acquired from a consenting physical human participant via live camera or verified recording.
        Eligible for inclusion in genuine research machine learning datasets upon passing audit.

    SIMULATED:
        Generated programmatically, via synthetic test fixtures, or demonstration generators.
        STRICTLY DISALLOWED from entering genuine human research ML training or evaluation splits.

    PIPELINE_VALIDATION:
        Controlled technical verification frames or calibration sequences used to test vision
        accuracy under deliberate synthetic/known physical conditions. Excluded from general ML
        by default unless explicitly authorized under documented protocols.

    UNKNOWN:
        Any recording or dataset lacking documented provenance, missing verified consent,
        or of unverified origin. Excluded by default from all genuine research ML pipelines.
    """
    REAL_HUMAN = "REAL_HUMAN"
    SIMULATED = "SIMULATED"
    PIPELINE_VALIDATION = "PIPELINE_VALIDATION"
    UNKNOWN = "UNKNOWN"


ALL_PROVENANCES = [p.value for p in DataProvenance]
PROVENANCE_SET: Set[str] = set(ALL_PROVENANCES)


def is_valid_provenance(prov: str) -> bool:
    """Validates if a provenance string is one of the four official designations."""
    return prov in PROVENANCE_SET


def is_eligible_for_genuine_ml(prov: str) -> bool:
    """Returns True ONLY for REAL_HUMAN provenance."""
    return prov == DataProvenance.REAL_HUMAN.value
