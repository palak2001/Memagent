"""M4 gate: fact extraction from exchange (LLM mocked)."""

from unittest.mock import MagicMock

import pytest

from neander.pipeline.extractor import extract_facts


def _mock_provider(json_response: dict):
    provider = MagicMock()
    provider.extract_json.return_value = json_response
    return provider


def test_facts_extracted_and_typed():
    provider = _mock_provider({
        "facts": [
            {"content": "The user prefers dark mode.", "category": "preference"},
            {"content": "The user works as a software engineer.", "category": "fact"},
        ]
    })
    facts = extract_facts(provider, "I love dark mode", "Got it!")
    assert len(facts) == 2
    contents = [f["content"] for f in facts]
    assert any("dark mode" in c for c in contents)
    categories = {f["category"] for f in facts}
    assert "preference" in categories
    assert "fact" in categories


def test_empty_response_returns_empty_list():
    provider = _mock_provider({"facts": []})
    facts = extract_facts(provider, "Hello", "Hi!")
    assert facts == []


def test_malformed_response_returns_empty_list():
    provider = MagicMock()
    provider.extract_json.return_value = {"unexpected": "schema"}
    facts = extract_facts(provider, "Test", "Test")
    assert facts == []


def test_exception_returns_empty_list():
    provider = MagicMock()
    provider.extract_json.side_effect = RuntimeError("network error")
    facts = extract_facts(provider, "Test", "Test")
    assert facts == []


def test_partial_facts_filtered():
    """Facts missing required fields are silently dropped."""
    provider = _mock_provider({
        "facts": [
            {"content": "Valid fact", "category": "fact"},
            {"only_content": "Missing category"},  # malformed
            {"only_category": "preference"},        # malformed
        ]
    })
    facts = extract_facts(provider, "test", "test")
    assert len(facts) == 1
    assert facts[0]["content"] == "Valid fact"
