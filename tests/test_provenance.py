"""Unit tests for data provenance enforcement and research consent verification."""

import pytest
from lecture_analyzer.collection.consent import verify_research_collection_eligibility
from lecture_analyzer.dataset.provenance import DataProvenance, is_eligible_for_genuine_ml, is_valid_provenance


def test_provenance_taxonomies():
    assert is_valid_provenance("REAL_HUMAN") is True
    assert is_valid_provenance("SIMULATED") is True
    assert is_valid_provenance("PIPELINE_VALIDATION") is True
    assert is_valid_provenance("UNKNOWN") is True
    assert is_valid_provenance("FABRICATED") is False


def test_ml_eligibility_strictness():
    assert is_eligible_for_genuine_ml("REAL_HUMAN") is True
    assert is_eligible_for_genuine_ml("SIMULATED") is False
    assert is_eligible_for_genuine_ml("PIPELINE_VALIDATION") is False
    assert is_eligible_for_genuine_ml("UNKNOWN") is False


def test_consent_verification_gate_success():
    res = verify_research_collection_eligibility(
        participant_code="P001",
        consent_confirmed=True,
        is_live_camera=True,
    )
    assert res.is_valid is True
    assert res.error_message is None


def test_consent_verification_fails_without_consent():
    res = verify_research_collection_eligibility(
        participant_code="P001",
        consent_confirmed=False,
        is_live_camera=True,
    )
    assert res.is_valid is False
    assert "consent must be acknowledged" in res.error_message.lower()


def test_consent_verification_fails_without_participant_code():
    res = verify_research_collection_eligibility(
        participant_code="",
        consent_confirmed=True,
        is_live_camera=True,
    )
    assert res.is_valid is False
    assert "participant code is required" in res.error_message.lower()


def test_consent_verification_fails_if_not_live_camera():
    res = verify_research_collection_eligibility(
        participant_code="P001",
        consent_confirmed=True,
        is_live_camera=False,
    )
    assert res.is_valid is False
    assert "active live physical camera" in res.error_message.lower()
