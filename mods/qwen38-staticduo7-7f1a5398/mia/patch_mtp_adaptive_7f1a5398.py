# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the measured variable-depth MTP overlays from the pinned image.

CPU-only. Does not import vLLM/torch, change the input tree, or start containers.
Original vLLM SPDX headers are preserved in generated files. Refuse unknown
sources instead of silently applying a stale overlay after an image update.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMPORT = 'from vllm.v1.core.sched.qwen_mtp_adaptive import ENABLED as QWEN_MTP_ADAPTIVE, runtime_depth\n'
TARGETS = {
    "qwen_scheduler_adaptive.py": "v1/core/sched/scheduler.py",
    "qwen_model_runner_adaptive.py": "v1/worker/gpu/model_runner.py",
    "qwen_autoregressive_adaptive.py": "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
    "qwen_mtp_speculator_adaptive.py": "v1/worker/gpu/spec_decode/mtp/speculator.py",
    "qwen_cudagraph_adaptive.py": "v1/worker/gpu/cudagraph_utils.py"
}
SOURCE_SHA256 = {
    "v1/core/sched/scheduler.py": "95267641af27c4509a36c68e5a182673fe44d98f4f4b4fe558dfd3d5cc90fb74",
    "v1/worker/gpu/model_runner.py": "c55e17516895686ea4b5ed94f1e30c6217e6d6c4bbe3488dc9f64b515163fa70",
    "v1/worker/gpu/spec_decode/autoregressive/speculator.py": "fb0fc18df67279485727f0975bdd55caf4b3ab4c09352d702c03ffc8ad38869a",
    "v1/worker/gpu/spec_decode/mtp/speculator.py": "926d6c9ffdb368c971647911a60493820d056ce236fcae9d94a42fa2e006255d",
    "v1/worker/gpu/cudagraph_utils.py": "0154ccf85e58d5e8c0308018fd843e379e43860f429fc931d4551f93c406771d",
}

def replace(src, old, new, count=1):
    actual = src.count(old)
    if actual != count:
        raise ValueError(f"Source anchor mismatch: expected {count}, got {actual}: {old[:100]!r}")
    return src.replace(old, new)


def build(source_root, out):
    source_root, out = Path(source_root).resolve(), Path(out).resolve()
    if out == source_root or source_root in out.parents:
        raise ValueError("Output must be outside the original source tree")
    for name, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unsupported vLLM source {name}: {actual}; expected {expected}")
    pending, manifest = {}, {}

    def save(name, source, text):
        ast.parse(text)
        pending[name] = text
        manifest[name] = dict(
            original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            patched_sha256=hashlib.sha256(text.encode()).hexdigest())

    p = source_root / 'v1/core/sched/scheduler.py'
    s = p.read_text(encoding='utf-8')
    s = replace(s, 'import time\n', 'import time\nfrom vllm.v1.core.sched.qwen_mtp_adaptive import create_controller\n')
    s = replace(s, '        self.dynamic_sd_lookup: list[int] | None = None\n',
                '        self.qwen_mtp_controller = create_controller(vllm_config)\n        self.dynamic_sd_lookup: list[int] | None = None\n')
    s = replace(s, '        scheduled_encoder_input_stats = None\n',
                '        if self.qwen_mtp_controller is not None:\n            num_spec_tokens_to_schedule = self.qwen_mtp_controller.choose(num_scheduled_tokens)\n\n        scheduled_encoder_input_stats = None\n')
    s = replace(s, '        sampled_token_ids = model_runner_output.sampled_token_ids\n',
                '        if self.qwen_mtp_controller is not None:\n            self.qwen_mtp_controller.observe(scheduler_output, model_runner_output)\n        sampled_token_ids = model_runner_output.sampled_token_ids\n')
    # Do not insert max-K speculative padding into requests joining variable-K batches.
    s = replace(s, '(self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)',
                '(self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None and self.qwen_mtp_controller is None)')
    save('qwen_scheduler_adaptive.py', p, s)

    p = source_root / 'v1/worker/gpu/model_runner.py'
    s = replace(p.read_text(encoding='utf-8'), 'import functools\n', 'import functools\n' + IMPORT)
    s = replace(s, '        if not dummy_run:\n            # Update the request states.\n',
                '        if not dummy_run:\n            if QWEN_MTP_ADAPTIVE and self.speculator is not None:\n                self.speculator._qwen_active_steps = runtime_depth(scheduler_output.num_spec_tokens_to_schedule, self.num_speculative_steps)\n            # Update the request states.\n')
    s = replace(s, '            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n',
                '            self.req_states.draft_tokens[input_batch.idx_mapping, :draft_tokens.shape[1]] = draft_tokens\n')
    s = replace(s, '                self.req_states.draft_tokens[input_batch.idx_mapping],\n',
                '                self.req_states.draft_tokens[input_batch.idx_mapping, :getattr(self.speculator, "_qwen_active_steps", self.num_speculative_steps)],\n')
    save('qwen_model_runner_adaptive.py', p, s)

    p = source_root / 'v1/worker/gpu/spec_decode/autoregressive/speculator.py'
    s = replace(p.read_text(encoding='utf-8'), 'from typing import Any\n', 'from typing import Any\n' + IMPORT)
    s = replace(s, '    def _configure_fused_multi_step_decode(self) -> None:\n',
                '    def _configure_fused_multi_step_decode(self) -> None:\n        if QWEN_MTP_ADAPTIVE:\n            self.use_fused_multi_step_decode = False\n            return\n')
    s = replace(s, '        if self.num_speculative_steps == 1:\n            # Early exit.\n',
                '        if getattr(self, "_qwen_active_steps", self.num_speculative_steps) == 1:\n            # Early exit.\n')
    s = replace(s, '        return self.draft_tokens[:num_reqs]\n',
                '        return self.draft_tokens[:num_reqs, :getattr(self, "_qwen_active_steps", self.num_speculative_steps)]\n')
    # Only the non-fused runtime loop; leave capture-time fused loop untouched.
    s = replace(s, '        for step in range(1, self.num_speculative_steps):\n            # Rebuild every step',
                '        for step in range(1, getattr(self, "_qwen_active_steps", self.num_speculative_steps)):\n            # Rebuild every step')
    save('qwen_autoregressive_adaptive.py', p, s)

    p = source_root / 'v1/worker/gpu/spec_decode/mtp/speculator.py'
    s = replace(p.read_text(encoding='utf-8'),
                '        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:\n',
                '        if self.share_mtp_topk_indices and getattr(self, "_qwen_active_steps", self.num_speculative_steps) > 1:\n')
    save('qwen_mtp_speculator_adaptive.py', p, s)

    p = source_root / 'v1/worker/gpu/cudagraph_utils.py'
    s = replace(p.read_text(encoding='utf-8'), 'from collections import defaultdict\n', 'from collections import defaultdict\n' + IMPORT)
    s = replace(s, '        capture_varlen_decode = (\n',
                '        if QWEN_MTP_ADAPTIVE and self.decode_query_len == self.vllm_config.num_speculative_tokens + 1:\n            decode_query_lens = list(range(2, self.decode_query_len + 1))\n\n        capture_varlen_decode = (\n')
    save('qwen_cudagraph_adaptive.py', p, s)

    controller = ROOT / 'mtp_adaptive/qwen_mtp_adaptive.py'
    save(controller.name, controller, controller.read_text(encoding='utf-8'))
    # Publish only after every source and every generated AST has passed.
    out.mkdir(parents=True, exist_ok=True)
    for name, text in pending.items():
        (out / name).write_text(text, encoding='utf-8', newline='\n')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                     encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-sources', action='store_true')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.list_sources:
        for output, target in TARGETS.items():
            print(f'{output}|{target}')
        return
    if args.source_root is None or args.output_dir is None:
        parser.error('--source-root and --output-dir are required')
    try:
        manifest = build(args.source_root, args.output_dir)
    except (OSError, ValueError, SyntaxError) as exc:
        parser.exit(1, f'adaptive MTP patch refused: {exc}\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
