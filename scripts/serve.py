#!/usr/bin/env python3
"""
Phase 8: start the chat server and web page.

Usage:
    python scripts/serve.py                  # then open http://127.0.0.1:8000
    python scripts/serve.py --port 9000
    python scripts/serve.py --host 0.0.0.0   # reachable from other machines: see the warning below

There is no login. By default the server only accepts connections from this computer. If you
use --host 0.0.0.0, anyone on the network can upload documents and ask questions, so only do
that on a network you trust.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lending_agent.config import get_settings  # noqa: E402


def main() -> int:
    api = get_settings().api
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=api.host, help=f"Address to listen on (default: {api.host})")
    parser.add_argument("--port", type=int, default=api.port, help=f"Port (default: {api.port})")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("✗ uvicorn is not installed. Run: pip install -r requirements.txt")
        return 1

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"⚠ Listening on {args.host}: there is no login, so anyone who can reach this port can use the agent.")
    print(f"→ Open http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port} in your browser. Ctrl+C stops the server.")
    uvicorn.run("lending_agent.api.app:create_default_app", factory=True, host=args.host, port=args.port,
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
