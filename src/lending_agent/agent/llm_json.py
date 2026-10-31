"""
Shared helper for reading JSON out of an LLM reply. Small local models often
wrap JSON in code fences or add a friendly sentence around it, so we tolerate
that -- but anything that still isn't a JSON object is rejected (returns None).
"""
from __future__ import annotations

import json
import re


def parse_llm_json(raw: str) -> dict | None:
    """Return the JSON object found in `raw`, or None if there isn't a usable one."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    for candidate in (text, _outermost_braces(text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _outermost_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else None
