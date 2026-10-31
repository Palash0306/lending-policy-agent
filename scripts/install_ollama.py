#!/usr/bin/env python3
"""
Optional convenience script: detects your OS and helps you install the
Ollama *application* itself (the local server that does the actual AI
inference). This is separate from `pip install -r requirements.txt`, which
only installs the small Python "messenger" package (`ollama`) that talks
to that server over HTTP -- it does not install the server.

This script NEVER runs anything with system-level effects (sudo, brew,
installers) without asking you to confirm first. If you'd rather do it
fully by hand, see the "Setup" section of README.md instead.

Usage:
    python scripts/install_ollama.py
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def _confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def _ollama_already_installed() -> bool:
    return shutil.which("ollama") is not None


def _run(cmd: str) -> int:
    print(f"\n$ {cmd}\n")
    return subprocess.call(cmd, shell=True)


def main() -> int:
    print("→ Checking whether Ollama is already installed...\n")

    if _ollama_already_installed():
        print("✓ 'ollama' was found on your PATH -- it looks like it's already installed.")
        print("  Next step: run `ollama serve` in a terminal (if it isn't already running),")
        print("  then `python scripts/verify_ollama.py` to confirm inference works.")
        return 0

    system = platform.system()  # 'Linux', 'Darwin' (macOS), or 'Windows'
    print(f"✗ Ollama not found on your PATH. Detected OS: {system}\n")

    if system == "Linux":
        print("Ollama publishes an official install script for Linux:")
        print("    curl -fsSL https://ollama.com/install.sh | sh\n")
        print("This downloads and runs a shell script from ollama.com with your user")
        print("permissions (it may ask for sudo internally to place the binary).")
        print("Review it yourself first if you'd like: https://ollama.com/install.sh\n")
        if _confirm("Run this install command now?"):
            code = _run("curl -fsSL https://ollama.com/install.sh | sh")
            if code == 0:
                print("\n✓ Install script finished. Next: run `ollama serve` in a terminal,")
                print("  then `python scripts/verify_ollama.py`.")
            else:
                print(f"\n✗ Install script exited with code {code}. See output above.")
            return code
        else:
            print("Skipped. Run the command yourself whenever you're ready, or download")
            print("directly from: https://ollama.com/download")
            return 1

    elif system == "Darwin":  # macOS
        has_brew = shutil.which("brew") is not None
        if has_brew:
            print("Homebrew is available. Recommended:\n    brew install ollama\n")
            if _confirm("Run 'brew install ollama' now?"):
                code = _run("brew install ollama")
                if code == 0:
                    print("\n✓ Installed. Next: run `ollama serve` in a terminal,")
                    print("  then `python scripts/verify_ollama.py`.")
                else:
                    print(f"\n✗ brew install exited with code {code}. See output above.")
                return code
            else:
                print("Skipped. Run it yourself whenever ready, or download the macOS app from:")
                print("https://ollama.com/download")
                return 1
        else:
            print("Homebrew not found. Easiest path on macOS without Homebrew:")
            print("  1. Download the app: https://ollama.com/download")
            print("  2. Open it once -- it installs the 'ollama' CLI and starts the server.")
            return 1

    elif system == "Windows":
        print("Ollama on Windows ships as a graphical installer (OllamaSetup.exe) --")
        print("this script can't run a GUI installer for you, and won't try to.")
        print("  1. Download it here: https://ollama.com/download/windows")
        print("  2. Run the installer (it sets up the 'ollama' command and starts the")
        print("     server automatically).")
        print("  3. Then come back and run: python scripts\\verify_ollama.py")
        return 1

    else:
        print(f"Unrecognized OS '{system}'. Please install manually from:")
        print("https://ollama.com/download")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
