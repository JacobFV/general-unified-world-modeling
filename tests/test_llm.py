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
import torch

from general_unified_world_model.llm.projection_builder import (
    llm_project,
    llm_build,
    LLMProjectionResult,
    _get_all_field_paths,
    _get_top_level_domains,
    _build_schema_description,
    _parse_llm_response,
    _source_to_profile,
    _load_dotenv,
)
from general_unified_world_model.training.heterogeneous import (
    DatasetSpec, DataSource, InputSpec, OutputSpec,
)
from general_unified_world_model.inference import GeneralUnifiedWorldModel


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
        result = LLMProjectionResult(include=["financial", "regime"], reasoning="test")
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
        assert "financial" in result.include
        assert "regime" in result.include
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

        assert "firm_ACME" in result.entities
        assert any("regime" in p for p in result.include)

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
        assert any("country" in p for p in result.include)
        assert "regime" in result.include


class TestErrorHandling:
    @patch("general_unified_world_model.llm.projection_builder._load_dotenv")
    def test_missing_api_key_raises(self, _mock_dotenv):
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
        assert result.include == ["*"]


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
        includes = result.include
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

        includes = result.include
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


# ── LLMProjectionResult.to_model tests ───────────────────────────────────


class TestLLMProjectionResultToModel:
    def test_to_model_returns_world_model(self):
        result = LLMProjectionResult(include=["financial.yield_curves", "regime"])
        model = result.to_model(d_model=32)
        assert isinstance(model, GeneralUnifiedWorldModel)
        assert len(model.bound.field_names) > 0

    def test_to_model_with_datasets(self):
        result = LLMProjectionResult(include=["regime"])
        spec = DatasetSpec(
            name="t",
            input_specs=[InputSpec(key="v", semantic_type="x", field_path="regime.growth_regime")],
        )
        ds = DataSource(spec=spec, data={"v": torch.randn(10)})
        model = result.to_model(datasets=[ds], d_model=32)
        assert model._dataset_specs.get("t") is not None


# ── _source_to_profile tests ─────────────────────────────────────────────


class TestSourceToProfile:
    def test_converts_datasource(self):
        spec = DatasetSpec(
            name="macro_test",
            description="My macro dataset",
            input_specs=[
                InputSpec(key="gdp", semantic_type="GDP", field_path="country_us.macro.output.gdp_nowcast"),
            ],
        )
        source = DataSource(spec=spec, data={"gdp": torch.randn(100)})
        profile = _source_to_profile(source)

        assert profile.name == "macro_test"
        assert profile.description == "My macro dataset"
        assert profile.n_samples == 100
        assert "gdp" in profile.columns
        assert len(profile.input_specs) == 1

    def test_handles_empty_data(self):
        spec = DatasetSpec(name="empty", input_specs=[])
        source = DataSource(spec=spec, data={})
        profile = _source_to_profile(source)
        assert profile.n_samples == 0

    def test_infers_frequency(self):
        spec = DatasetSpec(
            name="freq_test",
            input_specs=[
                InputSpec(key="v", semantic_type="x", field_path="regime.growth_regime", frequency=22),
            ],
        )
        source = DataSource(spec=spec, data={"v": torch.randn(50)})
        profile = _source_to_profile(source)
        assert profile.update_frequency == "monthly"


# ── llm_build tests (mocked) ─────────────────────────────────────────────


class TestLLMBuild:
    @patch("general_unified_world_model.llm.projection_builder.llm_project")
    def test_projection_only(self, mock_proj):
        """llm_build with no datasets returns untrained model."""
        mock_proj.return_value = LLMProjectionResult(
            include=["financial.yield_curves", "regime"], reasoning="test",
        )
        model = llm_build(
            "test", datasets=None, api_key="sk-test", d_model=32,
        )
        assert isinstance(model, GeneralUnifiedWorldModel)
        assert len(model.bound.field_names) > 0

    @patch("general_unified_world_model.llm.projection_builder.llm_project")
    def test_with_datasets_and_training(self, mock_proj):
        """llm_build with datasets should train the model."""
        mock_proj.return_value = LLMProjectionResult(
            include=["financial.yield_curves", "regime"], reasoning="test",
        )

        bound_for_spec = mock_proj.return_value.compile(T=1, d_model=32)
        field = bound_for_spec.field_names[0]

        spec = DatasetSpec(
            name="s",
            input_specs=[InputSpec(key="v", semantic_type="x", field_path=field)],
            output_specs=[OutputSpec(key="v", semantic_type="x", field_path=field)],
        )
        ds = DataSource(spec=spec, data={"v": torch.randn(50)})

        model = llm_build(
            "test",
            datasets=[ds],
            api_key="sk-test",
            n_steps=3,
            d_model=32,
            batch_size=4,
            log_every=100,
        )
        assert isinstance(model, GeneralUnifiedWorldModel)

    @patch("general_unified_world_model.llm.projection_builder.llm_project")
    def test_n_steps_zero_skips_training(self, mock_proj):
        """n_steps=0 returns an untrained model even with datasets."""
        mock_proj.return_value = LLMProjectionResult(
            include=["regime"], reasoning="test",
        )
        spec = DatasetSpec(name="s", input_specs=[])
        ds = DataSource(spec=spec, data={})

        model = llm_build(
            "test", datasets=[ds], api_key="sk-test",
            n_steps=0, d_model=32,
        )
        assert isinstance(model, GeneralUnifiedWorldModel)
