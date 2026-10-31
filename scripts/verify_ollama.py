#!/usr/bin/env python3
"""
Phase 1 entry point: confirm Ollama is running, the configured model is
pulled, and local inference actually works end to end.

Usage:
    python scripts/verify_ollama.py
    python scripts/verify_ollama.py --model llama3.1:8b
    python scripts/verify_ollama.py --host http://gpu-box:11434
    python scripts/verify_ollama.py --no-pull   # fail instead of auto-pulling
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lending_agent.config import OllamaSettings  # noqa: E402
from lending_agent.llm.ollama_client import HealthcheckResult, OllamaClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Override OLLAMA_MODEL for this run")
    parser.add_argument("--host", default=None, help="Override OLLAMA_HOST for this run")
    parser.add_argument("--no-pull", action="store_true", help="Don't auto-pull a missing model")
    args = parser.parse_args()

    base = OllamaSettings()
    settings = OllamaSettings(
        host=args.host or base.host,
        model=args.model or base.model,
        request_timeout=base.request_timeout,
        pull_timeout=base.pull_timeout,
    )

    client = OllamaClient(settings)

    print(f"→ Ollama host:   {settings.host}")
    print(f"→ Target model:  {settings.model}")
    print("→ Checking connection and model availability...\n")

    result = client.healthcheck(auto_pull=not args.no_pull)

    print(f"  Server reachable:   {'yes' if result.reachable else 'NO'}")
    print(f"  Model available:    {'yes' if result.model_available else 'NO'}")
    print(f"  Installed models:   {', '.join(result.installed_models) or '(none)'}")

    if result.error:
        print(f"\n✗ FAILED: {result.error}")
        _print_hint(result)
        return 1

    print(f"  Sample inference:   '{result.sample_response}'")

    if result.ok:
        print("\n✓ Phase 1 check passed — local inference is working end to end.")
        return 0

    print("\n✗ FAILED: healthcheck did not fully pass, see fields above.")
    return 1


def _print_hint(result: HealthcheckResult) -> None:
    if not result.reachable:
        print(
            "\nHint: is the Ollama server running? Start it with `ollama serve` "
            "in another terminal, or check OLLAMA_HOST in your .env."
        )
    elif not result.model_available:
        print(
            "\nHint: pull the model manually with `ollama pull <model>`, "
            "or rerun this script without --no-pull to auto-pull it."
        )


if __name__ == "__main__":
    raise SystemExit(main())
