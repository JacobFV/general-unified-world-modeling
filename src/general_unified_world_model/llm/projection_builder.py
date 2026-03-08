"""LLM-powered projection builder: describe your modeling needs in plain English.

Uses raw HTTP calls to the Anthropic or OpenAI API — no SDK dependency.
The LLM reads the full World schema field list and selects which paths,
firms, individuals, countries, and sectors to include.

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

from general_unified_world_model.projection.subset import WorldProjection, project
from general_unified_world_model.schema.world import World


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
    lines.append("  - firms: list of firm names (e.g. ['AAPL', 'NVDA'])")
    lines.append("  - individuals: list of individual names (e.g. ['fed_chair', 'ceo_nvda'])")
    lines.append("  - countries: additional country codes beyond us/cn/eu (e.g. ['jp', 'uk'])")
    lines.append("  - sectors: additional sector names (e.g. ['healthcare', 'industrials'])")
    lines.append("  - supply_chains: additional supply chain nodes (e.g. ['rare_earths'])")

    return "\n".join(lines)


# ── LLM API calls ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a world model schema designer. Given a user's description of their modeling needs, select the relevant parts of the General Unified World Model schema.

You must respond with a JSON object containing:
{
  "include": ["list", "of", "dotted.field.paths"],
  "exclude": [],
  "firms": ["list", "of", "firm", "names"],
  "individuals": ["list", "of", "individual", "names"],
  "countries": ["list", "of", "country", "codes"],
  "sectors": ["list", "of", "sector", "names"],
  "supply_chains": ["list", "of", "supply_chain", "names"],
  "reasoning": "Brief explanation of why these fields were selected"
}

Rules:
1. Use the EXACT dotted path names from the schema. You can include whole domains (e.g. "financial") or specific sub-paths (e.g. "financial.yield_curves").
2. Always include "regime" — it's the compressed world state that ties everything together.
3. Always include relevant "forecasts" sub-paths for the user's use case.
4. Include "events" if the user needs real-time awareness.
5. Include "trust" if the user needs epistemic calibration.
6. For firms, use ticker symbols or short names. For individuals, use role-based names.
7. Country codes: us, cn, eu, jp, uk, in, kr, br, ru, etc. The schema already includes us, cn, eu as defaults.
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


# ── Public API ──────────────────────────────────────────────────────────

@dataclass
class LLMProjectionResult:
    """Result from LLM-powered projection design.

    Contains the WorldProjection plus the LLM's reasoning.
    """
    projection: WorldProjection
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
        return project(self.projection, T=T, H=H, W=W, d_model=d_model)


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
        LLMProjectionResult with the designed WorldProjection.

    Example:
        result = llm_project("Model Apple's business in the tech sector")
        bound = result.compile(T=1, H=32, W=32, d_model=64)
        print(result.reasoning)
    """
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
    all_paths = set(_get_all_field_paths())
    validated_includes = []

    for path in parsed.get("include", []):
        # Accept if it's a valid top-level domain
        top = path.split(".")[0]
        if top in valid_domains:
            validated_includes.append(path)

    if not validated_includes:
        validated_includes = ["*"]  # Fallback to everything

    # Build projection
    projection = WorldProjection(
        include=validated_includes,
        exclude=parsed.get("exclude", []),
        firms=parsed.get("firms", []),
        individuals=parsed.get("individuals", []),
        countries=parsed.get("countries", []),
        sectors=parsed.get("sectors", []),
        supply_chains=parsed.get("supply_chains", []),
    )

    return LLMProjectionResult(
        projection=projection,
        reasoning=parsed.get("reasoning", ""),
        raw_response=parsed,
    )
