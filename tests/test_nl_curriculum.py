"""Tests for the curriculum specification system."""

import tempfile
from pathlib import Path

import pytest

from general_unified_world_model.training.dag_curriculum import (
    CurriculumSpec,
    Stage,
    StagesInParallel,
)


class TestStage:
    def test_defaults(self):
        s = Stage(description="test")
        assert s.datasets == []
        assert s.include is None
        assert s.H is None

    def test_explicit_include(self):
        s = Stage(description="test", include=["financial"])
        assert s.include == ["financial"]


class TestCurriculumSpec:
    def test_simple_spec(self):
        spec = CurriculumSpec(
            plan=[
                StagesInParallel(
                    name="stage1",
                    stages=[
                        Stage(
                            description="Financial markets",
                            include=["financial"],
                        ),
                        Stage(
                            description="Macroeconomics",
                            include=["country_us.macro"],
                        ),
                    ],
                ),
            ],
        )
        nodes = spec.to_training_nodes()
        assert len(nodes) == 2
        for n in nodes:
            assert n.parents == []

    def test_two_stage_spec(self):
        spec = CurriculumSpec(
            plan=[
                StagesInParallel(
                    name="basics",
                    stages=[
                        Stage(
                            description="Financial yield curves",
                            include=["financial.yield_curves"],
                        ),
                        Stage(
                            description="GDP and inflation",
                            include=["country_us.macro"],
                        ),
                    ],
                ),
                StagesInParallel(
                    name="advanced",
                    builds_on="basics",
                    stages=[
                        Stage(
                            description="How macro drives financial markets",
                            include=["financial", "country_us.macro", "regime"],
                        ),
                    ],
                ),
            ],
        )
        nodes = spec.to_training_nodes()
        assert len(nodes) == 3

        stage1_nodes = [n for n in nodes if n.parents == []]
        assert len(stage1_nodes) == 2

        stage2_nodes = [n for n in nodes if len(n.parents) > 0]
        assert len(stage2_nodes) == 1
        assert len(stage2_nodes[0].parents) == 2

    def test_explicit_include_on_nodes(self):
        spec = CurriculumSpec(
            plan=[
                StagesInParallel(
                    name="test",
                    stages=[
                        Stage(
                            description="Whatever text",
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
            plan=[
                StagesInParallel(
                    name="test",
                    stages=[
                        Stage(
                            description="Test",
                            include=["financial"],
                            H=64, n_layers=8, n_steps=5000,
                        ),
                    ],
                ),
            ],
        )
        nodes = spec.to_training_nodes()
        assert nodes[0].H == 64
        assert nodes[0].W == 32
        assert nodes[0].n_layers == 8
        assert nodes[0].n_steps == 5000

    def test_from_dict_new_format(self):
        d = {
            "name": "test_curriculum",
            "defaults": {"H": 24, "W": 24, "d_model": 32,
                         "n_layers": 2, "n_loops": 1, "n_steps": 100},
            "plan": [
                {
                    "name": "stage1",
                    "stages": [
                        {"description": "Financial markets", "include": ["financial"]},
                    ],
                },
                {
                    "name": "stage2",
                    "builds_on": "stage1",
                    "stages": [
                        {"description": "Macro and finance combined",
                         "include": ["financial", "country_us.macro", "regime"]},
                    ],
                },
            ],
        }
        spec = CurriculumSpec.from_dict(d)
        assert spec.name == "test_curriculum"
        assert len(spec.plan) == 2
        nodes = spec.to_training_nodes()
        assert len(nodes) == 2
        assert nodes[1].parents != []

    def test_no_include_defaults_to_wildcard(self):
        """Stages without explicit include default to ['*']."""
        spec = CurriculumSpec(
            plan=[
                StagesInParallel(
                    name="test",
                    stages=[
                        Stage(description="Everything"),
                    ],
                ),
            ],
        )
        nodes = spec.to_training_nodes()
        assert nodes[0].include == ["*"]

    def test_to_dict_roundtrip(self):
        """to_dict -> from_dict produces equivalent spec."""
        spec = CurriculumSpec(
            name="roundtrip_test",
            plan=[
                StagesInParallel(
                    name="foundations",
                    stages=[
                        Stage(description="Finance", include=["financial"], datasets=["yahoo"]),
                        Stage(description="Macro", include=["country_us.macro"]),
                    ],
                ),
                StagesInParallel(
                    name="integration",
                    builds_on="foundations",
                    stages=[
                        Stage(description="All together", include=["*"], n_layers=12),
                    ],
                ),
            ],
        )
        d = spec.to_dict()
        assert "plan" in d
        assert d["plan"][0]["stages"][0]["description"] == "Finance"

        spec2 = CurriculumSpec.from_dict(d)
        assert spec2.name == spec.name
        assert len(spec2.plan) == len(spec.plan)
        assert spec2.plan[0].stages[0].description == "Finance"
        assert spec2.plan[1].builds_on == "foundations"
