"""Preflight all source hashes and anchors before writing any changed source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def render(site, manifest):
    originals = {}
    for relative, expected in manifest["sha256"].items():
        data = (site / relative).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise SystemExit(f"source hash mismatch: {relative}: {actual} != {expected}")
        originals[relative] = data.decode("utf-8")
    changed = {}
    for relative, old, new, count in manifest["replacements"]:
        source = changed.get(relative, originals[relative])
        if source.count(old) != count:
            raise SystemExit(f"anchor mismatch: {relative}: {old!r}")
        changed[relative] = source.replace(old, new, count)
    for relative, source in changed.items():
        ast.parse(source, filename=relative)
    return changed


def main(script):
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--payload", required=True, type=Path)
    model = script.stem == "patch_vllm_qwen38next"
    if model:
        parser.add_argument("--direction", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(script.with_suffix(".json").read_text())
    changed = render(args.site, manifest)
    # Read payloads before changing source, so a missing input fails cleanly.
    if model:
        payloads = {
            "refusal_projection.py": (args.payload / "refusal_projection.py").read_bytes(),
            "refusal_direction_flashnext.npy": args.direction.read_bytes(),
        }
    else:
        payloads = {
            "v1/worker/gpu/refusal_utils.py": (args.payload / "refusal_utils.py").read_bytes(),
        }
    for relative, source in changed.items():
        (args.site / relative).write_text(source, encoding="utf-8")
    for relative, data in payloads.items():
        (args.site / relative).write_bytes(data)
    print(f"{script.stem}: {len(changed)} source files patched; strict hashes passed")
