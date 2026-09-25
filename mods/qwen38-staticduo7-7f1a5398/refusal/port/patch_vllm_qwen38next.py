#!/usr/bin/env python3
"""Strict refusal port for vLLM nightly 7f1a5398; CPU validated, GPU unvalidated."""
from pathlib import Path
from port_common import main

if __name__ == "__main__":
    main(Path(__file__))
