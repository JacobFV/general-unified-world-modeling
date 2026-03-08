"""Tests for the natural language curriculum specification system."""

import tempfile
from pathlib import Path

import pytest

from general_unified_world_model.training.dag_curriculum import (
    resolve_subject,
    CurriculumSpec,
    CurriculumStage,
    CurriculumSubject,
    STANDARD_CURRICULUM,
)


class TestResolveSubject:
    def test_financial(self):
        paths = resolve_subject("Core financial markets and yield curves")
        assert "financial" in paths or "financial.yield_curves" in paths

    def test_macro(self):
        paths = resolve_subject("US macroeconomic fundamentals")
        assert any("macro" in p for p in paths)

    def test_geopolitics(self):
        paths = resolve_subject("Geopolitical tensions between US and China")
        assert any("politics" in p for p in paths)

    def test_commodities(self):
        paths = resolve_subject("Oil prices and commodity markets")
        assert any("resources" in p or "energy" in p for p in paths)

    def test_technology(self):
        paths = resolve_subject("AI and semiconductor technology")
        assert any("technology" in p for p in paths)

    def test_always_includes_regime(self):
        """Non-trivial subjects should always include regime."""
        paths = resolve_subject("Financial markets and credit")
        assert "regime" in paths

    def test_no_match_fallback(self):
        """Completely unrecognized subject should fall back to wildcard."""
        paths = resolve_subject("zzzzzzz completely unknown gibberish")
        assert paths == ["*"]

    def test_multiple_domains(self):
        """Subject spanning multiple domains should include all."""
        paths = resolve_subject("How inflation affects equity markets and commodity prices")
        assert any("inflation" in p or "macro" in p for p in paths)
        assert any("equit" in p or "financial" in p for p in paths)
        assert any("resource" in p or "commodit" in p for p in paths)


class TestCurriculumSubject:
    def test_defaults(self):
        s = CurriculumSubject(subject="test")
        assert s.datasets == []
        assert s.firms == []
        assert s.include is None
        assert s.H is None

    def test_explicit_include(self):
        s = CurriculumSubject(subject="test", include=["financial"])
        assert s.include == ["financial"]


class TestCurriculumSpec:
    def test_simple_spec(self):
        spec = CurriculumSpec(
            stages=[
                CurriculumStage(
                    name="stage1",
                    parallel=[
                        CurriculumSubject(subject="Financial markets"),
                        CurriculumSubject(subject="Macroeconomics"),
                    ],
                ),
            ],
        )
        nodes = spec.to_training_nodes()
        assert len(nodes) == 2
        # Both should have no parents (first stage)
        for n in nodes:
            assert n.parents == []

    def test_two_stage_spec(self):
        spec = CurriculumSpec(
            stages=[
                CurriculumStage(
                    name="basics",
                    parallel=[
                        CurriculumSubject(subject="Financial yield curves"),
                        CurriculumSubject(subject="GDP and inflation"),
                    ],
                ),
                CurriculumStage(
                    name="advanced",
                    builds_on="basics",
                    parallel=[
                        CurriculumSubject(subject="How macro drives financial markets"),
                    ],
                ),
            ],
        )
        nodes = spec.to_training_nodes()
        assert len(nodes) == 3

        # First stage nodes have no parents
        stage1_nodes = [n for n in nodes if n.parents == []]
        assert len(stage1_nodes) == 2

        # Second stage node has parents from first stage
        stage2_nodes = [n for n in nodes if len(n.parents) > 0]
        assert len(stage2_nodes) == 1
        assert len(stage2_nodes[0].parents) == 2

    def test_explicit_include_overrides_nl(self):
        spec = CurriculumSpec(
            stages=[
                CurriculumStage(
                    name="test",
                    parallel=[
                        CurriculumSubject(
                            subject="Whatever text",
                            include=["financial", "regime"],
                        ),
                    ],
                ),
            ],
        )
        nodes = spec.to_training_nodes()
        assert nodes[0].include == ["financial", "regime"]

    def test_hyperparameter_overrides(self):
        spec = CurriculumSpec(
            defaults={"H": 32, "W": 32, "n_layers": 4, "n_loops": 2,
                       "n_steps": 1000, "d_model": 64},
            stages=[
                CurriculumStage(
                    name="test",
                    parallel=[
                        CurriculumSubject(
                            subject="Test",
                            H=64, n_layers=8, n_steps=5000,
                        ),
                    ],
                ),
            ],
        )
        nodes = spec.to_training_nodes()
        assert nodes[0].H == 64  # overridden
        assert nodes[0].W == 32  # default
        assert nodes[0].n_layers == 8  # overridden
        assert nodes[0].n_steps == 5000  # overridden

    def test_from_dict(self):
        d = {
            "name": "test_curriculum",
            "defaults": {"H": 24, "W": 24, "d_model": 32,
                         "n_layers": 2, "n_loops": 1, "n_steps": 100},
            "stages": [
                {
                    "name": "stage1",
                    "parallel": [
                        {"subject": "Financial markets"},
                    ],
                },
                {
                    "name": "stage2",
                    "builds_on": "stage1",
                    "parallel": [
                        {"subject": "Macro and finance combined"},
                    ],
                },
            ],
        }
        spec = CurriculumSpec.from_dict(d)
        assert spec.name == "test_curriculum"
        assert len(spec.stages) == 2
        nodes = spec.to_training_nodes()
        assert len(nodes) == 2
        assert nodes[1].parents != []

    def test_firms_and_countries(self):
        spec = CurriculumSpec(
            stages=[
                CurriculumStage(
                    name="test",
                    parallel=[
                        CurriculumSubject(
                            subject="Corporate analysis",
                            firms=["AAPL", "NVDA"],
                            countries=["jp"],
                        ),
                    ],
                ),
            ],
        )
        nodes = spec.to_training_nodes()
        assert nodes[0].firms == ["AAPL", "NVDA"]
        assert nodes[0].countries == ["jp"]


class TestStandardCurriculum:
    def test_has_stages(self):
        assert len(STANDARD_CURRICULUM.stages) == 4

    def test_generates_nodes(self):
        nodes = STANDARD_CURRICULUM.to_training_nodes()
        assert len(nodes) == 12

    def test_dag_structure(self):
        nodes = STANDARD_CURRICULUM.to_training_nodes()
        # Foundation nodes have no parents
        foundations = [n for n in nodes if "foundations" in n.name]
        for f in foundations:
            assert f.parents == []

        # Cross-domain nodes have parents from foundations
        cross = [n for n in nodes if "cross_domain" in n.name]
        for c in cross:
            assert len(c.parents) == len(foundations)

        # Integration should have parents from complex_dynamics
        integration = [n for n in nodes if "integration" in n.name]
        assert len(integration) == 1
        assert len(integration[0].parents) > 0

    def test_all_nodes_have_include(self):
        nodes = STANDARD_CURRICULUM.to_training_nodes()
        for n in nodes:
            assert len(n.include) > 0, f"Node {n.name} has empty include"
