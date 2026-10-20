"""Research collection consent verification and protocol guard."""

from dataclasses import dataclass
import re
from typing import Optional, Tuple


@dataclass
class ConsentVerificationResult:
    is_valid: bool
    error_message: Optional[str] = None


def verify_research_collection_eligibility(
    participant_code: Optional[str],
    consent_confirmed: bool,
    is_live_camera: bool,
) -> ConsentVerificationResult:
    """Verifies all mandatory prerequisites before research data collection can begin.

    Guarantees:
    1. An anonymous participant code must be provided (e.g. 'P001', 'SUBJ_04').
    2. Consent confirmation must be explicitly acknowledged by the operator/user.
    3. The collection source must be verified as an active live camera.
    """
    if not consent_confirmed:
        return ConsentVerificationResult(
            is_valid=False,
            error_message="Explicit participant consent must be acknowledged before starting research collection."
        )

    if not participant_code or not participant_code.strip():
        return ConsentVerificationResult(
            is_valid=False,
            error_message="A valid anonymous participant code is required for research collection."
        )

    code = participant_code.strip()
    # Ensure participant code does not contain personal identifying information (e.g. email or names)
    if "@" in code or len(code) > 32 or not re.match(r"^[A-Za-z0-9_\-]+$", code):
        return ConsentVerificationResult(
            is_valid=False,
            error_message="Participant code must be an anonymous alphanumeric identifier (e.g. 'P001', 'P_02')."
        )

    if not is_live_camera:
        return ConsentVerificationResult(
            is_valid=False,
            error_message="Research data collection requires an active live physical camera source."
        )

    return ConsentVerificationResult(is_valid=True)
