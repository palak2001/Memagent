"""M4 gate: credential and PII filtering."""

import pytest

from neander.pipeline.safety import filter_fact, is_credential, is_pii


# --- Credential detection (must be dropped entirely) ---

@pytest.mark.parametrize("text", [
    "sk-abc123def456ghi789jkl012mno345p",          # OpenAI-style key
    "sk-ant-api03-fake1234567890abcdefghijklmnopqrstuvwxyz",  # Anthropic key
    "AKIAIOSFODNN7EXAMPLE",                          # AWS key
    "123-45-6789",                                   # SSN
    "4111 1111 1111 1111",                           # Credit card (spaces)
    "4111-1111-1111-1111",                           # Credit card (dashes)
    "password=mysecretpass",                          # labelled password
    "api_key=someReallyLongSecretValue",              # labelled API key
    "secret=supersecretvalue123",                    # labelled secret
])
def test_credential_detected(text):
    assert is_credential(text), f"Should detect credential in: {text!r}"


@pytest.mark.parametrize("text", [
    "The user prefers dark mode.",
    "The user works at Acme Corp.",
    "The user is learning Python.",
])
def test_non_credential_not_flagged(text):
    assert not is_credential(text), f"Should NOT flag as credential: {text!r}"


# --- PII detection (stored but not prompted) ---

@pytest.mark.parametrize("text", [
    "My email is john.doe@example.com.",
    "Call me at 555-867-5309.",
    "Phone: (800) 555-1234",
])
def test_pii_detected(text):
    assert is_pii(text), f"Should detect PII in: {text!r}"


# --- filter_fact integration ---

def test_credential_fact_is_dropped():
    drop, tag_pii = filter_fact("sk-testkey123456789012345678901234567890")
    assert drop is True


def test_ssn_fact_is_dropped():
    drop, tag_pii = filter_fact("My SSN is 123-45-6789")
    assert drop is True


def test_email_fact_is_tagged_pii():
    drop, tag_pii = filter_fact("My email is user@example.com")
    assert drop is False
    assert tag_pii is True


def test_clean_fact_passes_through():
    drop, tag_pii = filter_fact("The user prefers Python over JavaScript.")
    assert drop is False
    assert tag_pii is False
