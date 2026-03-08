"""LLM-powered projection builder: describe your modeling needs in plain English.

Uses raw HTTP calls to the Anthropic or OpenAI API — no SDK dependency.
The LLM reads the full World schema field list and selects which paths
and entities to include.

Usage:
    from general_unified_world_model.llm import llm_project

    result = llm_project(
        "I'm a hedge fund PM. I need to model US macro, rates, credit,
         and two firms: Apple and NVIDIA. I care about recession risk
         and the Fed's next move.",
        provider="anthropic",
        api_key="sk-ant-...",
    )

    bound = result.compile(T=1, H=64, W=64, d_model=64)
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field as dc_field
from typing import Optional

from general_unified_world_model.projection.subset import project
from general_unified_world_model.schema.world import World


# ── .env loading ─────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """Load .env file into ``os.environ`` if present (no python-dotenv dep)."""
    from pathlib import Path

    for directory in [Path.cwd()] + list(Path.cwd().parents):
        env_file = directory / ".env"
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = val
            except (OSError, PermissionError):
                pass
            break


# ── Schema introspection ────────────────────────────────────────────────

def _get_all_field_paths() -> list[str]:
    """Walk the World schema and return all dotted field paths."""
    from canvas_engineering import Field

    paths = []

    def _walk(obj, prefix: str):
        if not dataclasses.is_dataclass(obj):
            return
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            full_path = f"{prefix}.{f.name}" if prefix else f.name
            if isinstance(val, Field):
                paths.append(full_path)
            elif dataclasses.is_dataclass(val):
                _walk(val, full_path)

    world = World()
    _walk(world, "")
    return paths


def _get_top_level_domains() -> list[str]:
    """Get the names of all top-level domains on World."""
    return [f.name for f in dataclasses.fields(World)]


def _build_schema_description() -> str:
    """Build a concise schema description for the LLM prompt."""
    domains = _get_top_level_domains()
    all_paths = _get_all_field_paths()

    # Group paths by top-level domain
    groups: dict[str, list[str]] = {}
    for path in all_paths:
        top = path.split(".")[0]
        groups.setdefault(top, []).append(path)

    lines = ["Available World Model domains and their fields:\n"]
    for domain in domains:
        domain_paths = groups.get(domain, [])
        lines.append(f"## {domain} ({len(domain_paths)} fields)")
        # Show first 10 fields as examples, then summarize
        for p in domain_paths[:10]:
            lines.append(f"  - {p}")
        if len(domain_paths) > 10:
            lines.append(f"  ... and {len(domain_paths) - 10} more fields")
        lines.append("")

    lines.append("\nDynamic entities (add as many as needed):")
    lines.append("  Provide as a dict mapping names to types.")
    lines.append("  Available types: Business, Individual, Country, Sector, SupplyChainNode")
    lines.append("  Example: {\"firm_AAPL\": \"Business\", \"country_jp\": \"Country\", \"person_ceo\": \"Individual\"}")

    return "\n".join(lines)


# ── LLM API calls ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a world model schema designer. Given a user's description of their modeling needs, select the relevant parts of the General Unified World Model schema.

You must respond with a JSON object containing:
{
  "include": ["list", "of", "dotted.field.paths"],
  "exclude": [],
  "entities": {"firm_AAPL": "Business", "country_jp": "Country", "person_ceo": "Individual"},
  "reasoning": "Brief explanation of why these fields were selected"
}

Rules:
1. Use the EXACT dotted path names from the schema. You can include whole domains (e.g. "financial") or specific sub-paths (e.g. "financial.yield_curves").
2. Always include "regime" — it's the compressed world state that ties everything together.
3. Always include relevant "forecasts" sub-paths for the user's use case.
4. Include "events" if the user needs real-time awareness.
5. Include "trust" if the user needs epistemic calibration.
6. Entity names should use prefixes: "firm_" for Business, "person_" for Individual, "country_" for Country, "sector_" for Sector, "sc_" for SupplyChainNode.
7. Available entity types: Business, Individual, Country, Sector, SupplyChainNode.
8. Respond ONLY with the JSON object. No markdown code fences, no extra text."""


def _call_anthropic(prompt: str, api_key: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call the Anthropic Messages API via HTTP."""
    schema_desc = _build_schema_description()

    body = json.dumps({
        "model": model,
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT + "\n\n" + schema_desc,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Extract text from content blocks
    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block["text"]
    return text


def _call_openai(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Call the OpenAI Chat Completions API via HTTP."""
    schema_desc = _build_schema_description()

    body = json.dumps({
        "model": model,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + schema_desc},
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"]


def _parse_llm_response(text: str) -> dict:
    """Parse the LLM's JSON response, handling common formatting issues."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    return json.loads(text)


def _resolve_entities_from_response(parsed: dict) -> dict:
    """Resolve entity specifications from LLM response to instances.

    Handles the generic entities dict format:
        {"firm_AAPL": "Business", "country_jp": "Country"}
    """
    from general_unified_world_model.training.dag_curriculum import _resolve_entities

    entities_spec = parsed.get("entities", {})
    if not entities_spec:
        return {}

    return _resolve_entities(entities_spec)


# ── Public API ──────────────────────────────────────────────────────────

@dataclass
class LLMProjectionResult:
    """Result from LLM-powered projection design.

    Contains the include/exclude/entities selected by the LLM plus reasoning.
    """
    include: list = dc_field(default_factory=lambda: ["*"])
    exclude: list = dc_field(default_factory=list)
    entities: dict = dc_field(default_factory=dict)
    reasoning: str = ""
    raw_response: dict = dc_field(default_factory=dict)

    def compile(
        self,
        T: int = 1,
        H: int = 64,
        W: int = 64,
        d_model: int = 64,
    ):
        """Compile this projection to a BoundSchema."""
        return project(
            include=self.include, exclude=self.exclude,
            entities=self.entities, T=T, H=H, W=W, d_model=d_model,
        )

    def to_model(
        self,
        datasets: list | None = None,
        T: int = 1,
        d_model: int = 64,
        device: str = "cpu",
        **kwargs,
    ):
        """Create a ``GeneralUnifiedWorldModel`` from this projection.

        Convenience that skips the compile → WorldModel two-step.

        Args:
            datasets: Optional DataSource list to register on the model.
            T: Temporal extent.
            d_model: Latent dimension.
            device: Device.
            **kwargs: Forwarded to ``GeneralUnifiedWorldModel``.

        Returns:
            ``GeneralUnifiedWorldModel`` ready for inference or fine-tuning.
        """
        from general_unified_world_model.inference import GeneralUnifiedWorldModel

        return GeneralUnifiedWorldModel(
            include=self.include,
            exclude=self.exclude,
            entities=self.entities,
            T=T,
            d_model=d_model,
            device=device,
            data_sources=datasets,
            **kwargs,
        )


def llm_project(
    description: str,
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
) -> LLMProjectionResult:
    """Design a world model projection from a natural language description.

    Args:
        description: Plain English description of modeling needs.
            E.g. "I'm a macro strategist. I need US and EU macro,
                  rates, credit spreads, and recession probability."
        provider: "anthropic" or "openai".
        api_key: API key. If None, reads from ANTHROPIC_API_KEY or
            OPENAI_API_KEY environment variable.
        model: Model to use. Defaults to claude-sonnet-4-20250514 or gpt-4o-mini.

    Returns:
        LLMProjectionResult with include/exclude/entities and reasoning.

    Example:
        result = llm_project("Model Apple's business in the tech sector")
        bound = result.compile(T=1, H=32, W=32, d_model=64)
        print(result.reasoning)
    """
    # Load .env before resolving keys
    _load_dotenv()

    # Resolve API key
    if api_key is None:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"No API key provided. Set {provider.upper()}_API_KEY env var "
                f"or pass api_key= parameter."
            )

    # Call the LLM
    if provider == "anthropic":
        raw_text = _call_anthropic(description, api_key, model=model or "claude-sonnet-4-20250514")
    elif provider == "openai":
        raw_text = _call_openai(description, api_key, model=model or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")

    # Parse the response
    parsed = _parse_llm_response(raw_text)

    # Validate include paths against the schema
    valid_domains = set(_get_top_level_domains())
    validated_includes = []

    for path in parsed.get("include", []):
        # Accept if it's a valid top-level domain
        top = path.split(".")[0]
        if top in valid_domains:
            validated_includes.append(path)

    if not validated_includes:
        validated_includes = ["*"]  # Fallback to everything

    # Resolve entities
    entities = _resolve_entities_from_response(parsed)

    return LLMProjectionResult(
        include=validated_includes,
        exclude=parsed.get("exclude", []),
        entities=entities,
        reasoning=parsed.get("reasoning", ""),
        raw_response=parsed,
    )


# ── DataSource → DatasetProfile conversion ─────────────────────────────

def _source_to_profile(source) -> "DatasetProfile":
    """Convert a ``DataSource`` to a ``DatasetProfile`` for LLM consumption.

    Args:
        source: A DataSource (spec + data dict).

    Returns:
        DatasetProfile with metadata derived from the source.
    """
    from general_unified_world_model.training.dag_curriculum import DatasetProfile

    spec = source.spec
    data = source.data

    n_samples = 0
    columns: list[str] = []
    if isinstance(data, dict):
        columns = list(data.keys())
        for v in data.values():
            if hasattr(v, "shape"):
                n_samples = max(n_samples, v.shape[0] if v.dim() >= 1 else 1)
            elif hasattr(v, "__len__"):
                n_samples = max(n_samples, len(v))
            break

    update_freq = "unknown"
    for ispec in spec.input_specs:
        if ispec.frequency:
            f = ispec.frequency
            if f == 1:
                update_freq = "daily"
            elif f <= 5:
                update_freq = "weekly"
            elif f <= 22:
                update_freq = "monthly"
            elif f <= 66:
                update_freq = "quarterly"
            else:
                update_freq = "annual"
            break

    return DatasetProfile(
        name=spec.name,
        description=spec.description or f"Dataset: {spec.name}",
        input_specs=spec.input_specs,
        n_samples=n_samples,
        columns=columns,
        temporal_range=f"{n_samples} samples" if n_samples else "unknown",
        update_frequency=update_freq,
        source="local",
    )


# ── llm_build: combined projection + training ──────────────────────────

def llm_build(
    description: str,
    datasets: list | None = None,
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    device: str = "cpu",
    d_model: int = 64,
    n_steps: int = 500,
    lr: float = 1e-4,
    batch_size: int = 16,
    freeze_backbone: bool = False,
    full_curriculum: bool = False,
    checkpoint_dir: str = "checkpoints",
    log_every: int = 100,
):
    """Design and optionally train a world model from natural language.

    The primary high-level entry point.  Pass a plain-English description
    of what you want to model **plus** any datasets you want to train on.
    The LLM selects which of the 857 World schema fields to include; if
    you also pass datasets the model is trained on them via masked canvas
    learning — each dataset contributes gradient only for its fields while
    the shared regime latent integrates everything.

    Dynamic fidelity is automatic: domains you mention get full sub-field
    detail (high fidelity) while domains you only hint at are included at
    the coarse-grained summary level (low fidelity).

    For fine-tuning a pre-trained model on private data, see
    ``WorldModel.finetune()``.

    Args:
        description: What you want to model, in plain English.
        datasets: List of ``DataSource`` objects.  Each pairs a
            ``DatasetSpec`` (field mappings) with actual tensor data.
            If ``None``, an untrained model is returned (projection only).
        provider: ``"anthropic"`` or ``"openai"``.
        api_key: API key.  If ``None``, reads from the environment or
            ``.env`` file.
        model: LLM model override.
        device: ``"cpu"`` or ``"cuda"``.
        d_model: Latent dimension for canvas positions.
        n_steps: Training steps.  0 = skip training.
        lr: Learning rate.
        batch_size: Training batch size.
        freeze_backbone: If ``True``, only train encoder/decoder heads.
        full_curriculum: If ``True``, use the full LLM-designed DAG
            curriculum (two LLM calls; slower but better for large
            multi-domain datasets).
        checkpoint_dir: Checkpoint directory (full_curriculum mode).
        log_every: Print loss every N steps.

    Returns:
        ``GeneralUnifiedWorldModel`` trained on your datasets and ready
        for observe/predict.

    Example::

        import torch
        from general_unified_world_model import (
            llm_build, DataSource, DatasetSpec, InputSpec,
        )

        spec = DatasetSpec(
            name="my_data",
            description="Daily prices and macro",
            input_specs=[
                InputSpec(key="spx", semantic_type="S&P 500 level",
                          field_path="financial.equities.large_cap"),
                InputSpec(key="gdp", semantic_type="US GDP growth",
                          field_path="country_us.macro.output.gdp_nowcast"),
            ],
        )
        source = DataSource(spec=spec, data={
            "spx": torch.randn(252),
            "gdp": torch.randn(252),
        })

        model = llm_build(
            "Macro strategist: US equities + GDP dynamics",
            datasets=[source], n_steps=200,
        )
        model.observe("financial.equities.large_cap", 4500.0)
        predictions = model.predict()
    """
    from general_unified_world_model.inference import GeneralUnifiedWorldModel

    _load_dotenv()

    # Resolve API key
    if api_key is None:
        env_var = (
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        )
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(
                f"No API key. Set {env_var} in your environment or .env "
                f"file, or pass api_key=."
            )

    # Step 1 — LLM designs the schema projection
    print("[llm_build] Designing schema projection ...")
    proj_result = llm_project(
        description, provider=provider, api_key=api_key, model=model,
    )
    print(f"[llm_build] Domains: {proj_result.include}")
    if proj_result.entities:
        print(f"[llm_build] Entities: {list(proj_result.entities.keys())}")
    print(f"[llm_build] Reasoning: {proj_result.reasoning}")

    # Full DAG curriculum path (two LLM calls)
    if full_curriculum and datasets:
        return _llm_build_full_curriculum(
            description=description,
            datasets=datasets,
            proj_result=proj_result,
            provider=provider,
            api_key=api_key,
            model=model,
            device=device,
            d_model=d_model,
            checkpoint_dir=checkpoint_dir,
        )

    # Simple path: create model then finetune
    print("[llm_build] Building world model ...")
    world_model = GeneralUnifiedWorldModel(
        include=proj_result.include,
        exclude=proj_result.exclude,
        entities=proj_result.entities,
        d_model=d_model,
        device=device,
        data_sources=datasets,
    )

    n_fields = len(world_model.bound.field_names)
    n_pos = world_model.bound.layout.num_positions
    n_params = sum(p.numel() for p in world_model.backbone.parameters())
    print(f"[llm_build] {n_fields} fields, {n_pos} positions, "
          f"{n_params:,} backbone params")

    if datasets and n_steps > 0:
        print(f"[llm_build] Training on {len(datasets)} dataset(s) "
              f"for {n_steps} steps ...")
        world_model.finetune(
            datasets=datasets,
            n_steps=n_steps,
            lr=lr,
            freeze_backbone=freeze_backbone,
            batch_size=batch_size,
            log_every=log_every,
        )
    else:
        print("[llm_build] No training requested — returning projection-only model.")

    print("[llm_build] Done.")
    return world_model


def _llm_build_full_curriculum(
    description: str,
    datasets: list,
    proj_result: "LLMProjectionResult",
    provider: str,
    api_key: str,
    model: str | None,
    device: str,
    d_model: int,
    checkpoint_dir: str,
):
    """Internal: full LLM-designed DAG curriculum path."""
    from general_unified_world_model.training.dag_curriculum import (
        DAGCurriculumTrainer, build_curriculum,
    )

    profiles = [_source_to_profile(ds) for ds in datasets]

    print(f"[llm_build] Designing curriculum for {len(profiles)} dataset(s) ...")
    curriculum = build_curriculum(
        goal=description,
        datasets=profiles,
        provider=provider,
        api_key=api_key,
        model=model,
    )
    n_groups = len(curriculum.plan)
    n_nodes = sum(len(g.stages) for g in curriculum.plan)
    print(f"[llm_build] Curriculum: {n_groups} groups, {n_nodes} stages")

    source_map = {ds.spec.name: ds for ds in datasets}
    nodes = curriculum.to_training_nodes()

    trainer = DAGCurriculumTrainer(
        nodes=nodes,
        data_sources=source_map,
        checkpoint_dir=checkpoint_dir,
        device=device,
        backbone="scratch",
    )
    trainer.run()

    final_wm = trainer.get_final_model(device=device)
    for ds in datasets:
        final_wm.add_data(ds)

    print("[llm_build] Done.")
    return final_wm
