#!/usr/bin/env python3
"""Install precomposed overlays only on the exact vLLM image they target."""

import ast
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
SITE = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
MANIFEST = json.loads((BUNDLE / "manifest.json").read_text())
MARKER = Path("/opt/staticduo7-installed.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = MANIFEST["files"]
    if MARKER.exists():
        if MARKER.read_text() != (BUNDLE / "manifest.json").read_text():
            raise RuntimeError("staticduo7 marker belongs to a different bundle")
        for rel, hashes in files.items():
            if digest(SITE / rel) != hashes["overlay_sha256"]:
                raise RuntimeError(f"staticduo7 installed file changed: {rel}")
        print("staticduo7 overlays already installed and verified")
        return

    # Verify every source before changing the package: a newer image must fail closed.
    for rel, hashes in files.items():
        source = SITE / rel
        expected = hashes["base_sha256"]
        if expected is None:
            if source.exists():
                raise RuntimeError(f"staticduo7 expected absent source: {rel}")
        elif not source.is_file() or digest(source) != expected:
            raise RuntimeError(f"staticduo7 unexpected vLLM source: {rel}")
        overlay = BUNDLE / "overlays" / rel
        if digest(overlay) != hashes["overlay_sha256"]:
            raise RuntimeError(f"staticduo7 bundled overlay changed: {rel}")
        ast.parse(overlay.read_text(), filename=rel)

    for rel in files:
        source = BUNDLE / "overlays" / rel
        target = SITE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    refusal_root = Path("/opt/refusal")
    refusal_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        BUNDLE / "refusal/refusal_direction_flashnext.npy",
        refusal_root / "refusal_direction_flashnext.npy",
    )
    mia_root = Path("/opt/mia-staticduo7")
    mia_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        BUNDLE / "mia/draft_vocab_en_code_47k.txt",
        mia_root / "draft_vocab_en_code_47k.txt",
    )
    MARKER.write_text((BUNDLE / "manifest.json").read_text())
    print("staticduo7 exact-image overlays installed")


if __name__ == "__main__":
    main()
