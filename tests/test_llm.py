"""Tests for the LLM-powered projection builder.

These tests are intentionally flexible about exact field selections
since different LLM calls may produce reasonable but different results.
The tests verify structural correctness and that sensible domains are chosen.
"""

import json
import os
import unittest.mock as mock
from unittest.mock import patch, MagicMock

import pytest

from general_unified_world_model.llm.projection_builder import (
    llm_project,
    LLMProjectionResult,
    _get_all_field_paths,
    _get_top_level_domains,
    _build_schema_description,
    _parse_llm_response,
)
from general_unified_world_model.projection.subset import WorldProjection


# ── Unit tests (no API calls) ──────────────────────────────────────────


class TestSchemaIntrospection:
    def test_get_all_field_paths(self):
        paths = _get_all_field_paths()
        assert len(paths) > 100  # Should have hundreds of fields
        assert all("." in p for p in paths)  # All are dotted paths

    def test_get_top_level_domains(self):
        domains = _get_top_level_domains()
        assert "financial" in domains
        assert "regime" in domains
        assert "forecasts" in domains
        assert len(domains) >= 10

    def test_build_schema_description(self):
        desc = _build_schema_description()
        assert "financial" in desc
        assert "regime" in desc
        assert "Business" in desc
        assert len(desc) > 500  # Should be a substantial description


class TestParseResponse:
    def test_parses_clean_json(self):
        text = json.dumps({
            "include": ["financial", "regime"],
            "entities": {"firm_AAPL": "Business"},
            "reasoning": "test",
        })
        result = _parse_llm_response(text)
        assert result["include"] == ["financial", "regime"]

    def test_parses_json_with_code_fences(self):
        text = '```json\n{"include": ["financial"], "reasoning": "test"}\n```'
        result = _parse_llm_response(text)
        assert result["include"] == ["financial"]

    def test_parses_json_with_bare_fences(self):
        text = '```\n{"include": ["regime"], "reasoning": "test"}\n```'
        result = _parse_llm_response(text)
        assert result["include"] == ["regime"]


class TestLLMProjectionResult:
    def test_compile(self):
        proj = WorldProjection(include=["financial", "regime"])
        result = LLMProjectionResult(projection=proj, reasoning="test")
        bound = result.compile(T=1, H=24, W=24, d_model=32)
        assert len(bound.field_names) > 0


# ── Mock tests (simulate API responses) ────────────────────────────────


def _make_anthropic_response(content: dict) -> dict:
    """Build a fake Anthropic Messages API response."""
    return {
        "content": [{"type": "text", "text": json.dumps(content)}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }


def _make_openai_response(content: dict) -> dict:
    """Build a fake OpenAI Chat Completions API response."""
    return {
        "choices": [{"message": {"content": json.dumps(content)}}],
    }


HEDGE_FUND_RESPONSE = {
    "include": [
        "financial",
        "country_us.macro",
        "regime",
        "forecasts.macro",
        "forecasts.financial",
    ],
    "exclude": [],
    "entities": {"firm_AAPL": "Business", "firm_NVDA": "Business"},
    "reasoning": "Hedge fund needs financial markets, US macro, and forecasts.",
}

CEO_RESPONSE = {
    "include": [
        "country_us.macro",
        "sector_tech",
        "financial.equities",
        "regime",
        "forecasts",
    ],
    "exclude": [],
    "entities": {"firm_ACME": "Business", "person_ceo": "Individual", "person_cfo": "Individual"},
    "reasoning": "CEO needs company context, sector, and macro outlook.",
}

GEOPOLITICAL_RESPONSE = {
    "include": [
        "country_us",
        "country_cn",
        "country_eu",
        "regime",
        "events",
        "forecasts.geopolitical",
        "trust",
    ],
    "exclude": [],
    "entities": {"person_fed_chair": "Individual", "country_jp": "Country", "country_uk": "Country"},
    "reasoning": "Geopolitical analyst needs multi-country view.",
}


class TestMockedAnthropicCalls:
    @patch("urllib.request.urlopen")
    def test_hedge_fund_projection(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _make_anthropic_response(HEDGE_FUND_RESPONSE)
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = llm_project(
            "I'm a hedge fund PM trading US macro and rates with AAPL and NVDA positions.",
            provider="anthropic",
            api_key="sk-ant-test-key",
        )

        assert isinstance(result, LLMProjectionResult)
        assert isinstance(result.projection, WorldProjection)
        assert "financial" in result.projection.include
        assert "regime" in result.projection.include
        assert len(result.reasoning) > 0

    @patch("urllib.request.urlopen")
    def test_ceo_projection(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _make_anthropic_response(CEO_RESPONSE)
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = llm_project(
            "I'm a tech CEO. Model my company ACME in the context of the US economy.",
            provider="anthropic",
            api_key="sk-ant-test-key",
        )

        assert "firm_ACME" in result.projection.entities
        assert any("regime" in p for p in result.projection.include)

    @patch("urllib.request.urlopen")
    def test_result_compiles(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _make_anthropic_response(HEDGE_FUND_RESPONSE)
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = llm_project(
            "Hedge fund model",
            provider="anthropic",
            api_key="sk-ant-test-key",
        )

        bound = result.compile(T=1, H=64, W=64, d_model=32)
        assert len(bound.field_names) > 20


class TestMockedOpenAICalls:
    @patch("urllib.request.urlopen")
    def test_openai_projection(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _make_openai_response(GEOPOLITICAL_RESPONSE)
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = llm_project(
            "Geopolitical analyst covering US-China-EU dynamics.",
            provider="openai",
            api_key="sk-test-key",
        )

        assert isinstance(result, LLMProjectionResult)
        assert any("country" in p for p in result.projection.include)
        assert "regime" in result.projection.include


class TestErrorHandling:
    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No API key"):
                llm_project("test", provider="anthropic")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            llm_project("test", provider="gemini", api_key="key")

    @patch("urllib.request.urlopen")
    def test_fallback_on_empty_include(self, mock_urlopen):
        bad_response = {
            "include": ["nonexistent_domain.foo"],
            "reasoning": "oops",
        }
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            _make_anthropic_response(bad_response)
        ).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = llm_project("test", provider="anthropic", api_key="key")
        # Should fall back to ["*"] when no valid domains found
        assert result.projection.include == ["*"]


# ── Live integration tests (requires API keys) ─────────────────────────


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestLiveAnthropic:
    def test_hedge_fund_live(self):
        result = llm_project(
            "I'm a hedge fund PM. I need US macro, rates, credit, "
            "and positions in Apple and NVIDIA. I care about recession risk.",
        )

        # Flexible assertions: the LLM should pick reasonable domains
        includes = result.projection.include
        include_str = " ".join(includes)

        # Must include financial-related fields
        assert any(
            term in include_str
            for term in ["financial", "yield", "credit", "equities"]
        ), f"Expected financial domains, got: {includes}"

        # Must include regime (per system prompt instructions)
        assert any(
            "regime" in p for p in includes
        ), f"Expected regime, got: {includes}"

        # Should mention macro somewhere
        assert any(
            "macro" in p for p in includes
        ), f"Expected macro-related fields, got: {includes}"

        # Should compile without errors (large canvas for LLM-generated projections)
        bound = result.compile(T=1, H=96, W=96, d_model=32)
        assert len(bound.field_names) > 10

        # Reasoning should be non-empty
        assert len(result.reasoning) > 10


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestLiveOpenAI:
    def test_macro_strategist_live(self):
        result = llm_project(
            "I'm a macro strategist covering US, EU, and China. "
            "I need GDP, inflation, central bank policy, and regime indicators.",
            provider="openai",
        )

        includes = result.projection.include
        include_str = " ".join(includes)

        # Should include country-level macro
        assert any(
            "country" in p or "macro" in p for p in includes
        ), f"Expected country/macro fields, got: {includes}"

        # Should include regime
        assert any(
            "regime" in p for p in includes
        ), f"Expected regime, got: {includes}"

        # Should compile
        bound = result.compile(T=1, H=32, W=32, d_model=32)
        assert len(bound.field_names) > 10
