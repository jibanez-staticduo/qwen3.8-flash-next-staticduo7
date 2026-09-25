import importlib.util, hashlib, json
from pathlib import Path
SRC=Path('/tmp/vllm-nightly-7f1a5398/vllm')
BASE=Path('/home/staticduo/recipes/mods/qwen38-staticduo4-2a02f6ef/refusal')
OUT=Path('/tmp/staticduo7-astra')
for name,funcs in [('patch_vllm_qwen38next',['patch_model','patch_mtp']),('patch_vllm_runtime',['patch_scheduler','patch_model_runner','patch_worker','patch_kv_cache','patch_serve'])]:
 spec=importlib.util.spec_from_file_location(name,BASE/(name+'.py')); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
 ops=[]
 def capture(p,old,new,*,count=1):
  if 'hidden_states = hidden_states + self.ple(' in old: return
  if 'draft_tokens = self.speculator.propose(' in old: return
  if 'key="mtp.fc_embedding+fc_hidden"' in new:
   new=new.replace('key="mtp.fc_embedding+fc_hidden"','key="mtp.fc_hidden"',1)
   new=new.replace('key="mtp.fc_embedding+fc_hidden"','key="mtp.fc_embedding"',1)
  if 'prefill_token_ids=prefill_token_ids,' in old:
   old=old.replace('            prefill_token_ids=prefill_token_ids,\n','            prefill_token_ids=prefill_token_ids,\n            replay_start=request.replay_start,\n')
   new=new.replace('            prefill_token_ids=prefill_token_ids,\n','            prefill_token_ids=prefill_token_ids,\n            replay_start=request.replay_start,\n')
  if 'if not skip_attn_for_dummy_run:' in old:
   old=old.replace('if not skip_attn_for_dummy_run:', 'if skip_attn_for_dummy_run:')
   new=new.replace('if not skip_attn_for_dummy_run:', 'if skip_attn_for_dummy_run:')
  ops.append((str(p.relative_to(SRC)),old,new,count))
 mod.replace=capture
 for f in funcs:getattr(mod,f)(SRC)
 if name=='patch_vllm_runtime':
  draft='v1/worker/gpu/spec_decode/autoregressive/speculator.py'
  ops += [
   (draft,'import torch\n','import torch\nfrom vllm import refusal_projection\n',1),
   (draft,'        self.on_prefill_begin(num_reqs)\n',
'''        if refusal_projection.is_enabled():
            if self.pcp_manager is not None:
                raise RuntimeError("refusal per-request draft does not support PCP")
            if dummy_run:
                refusal_projection.fill_neutral(refusal_projection.get_lambda())
            else:
                # Step zero retains target row ranges, including rejected padding.
                refusal_projection.fill_target(
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                    refusal_projection.get_lambda(),
                    num_tokens=input_batch.num_tokens,
                )
        self.on_prefill_begin(num_reqs)
''',1),
   (draft,'        self.on_multi_step_decode_begin(num_reqs)\n',
'''        if refusal_projection.is_enabled():
            if dummy_run:
                refusal_projection.fill_neutral(refusal_projection.get_lambda())
            else:
                # Steps one onward use one row per request in the same order.
                # Fill outside both single-step and fused FULL graph replays.
                refusal_projection.fill_target(
                    input_batch.idx_mapping_np,
                    [1] * num_reqs,
                    refusal_projection.get_lambda(),
                    num_tokens=num_reqs,
                )
        self.on_multi_step_decode_begin(num_reqs)
''',1)]
 if name=='patch_vllm_qwen38next':
  ops += [
   ('models/qwen4_exp/nvidia/mtp.py',
'''        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )
''',
'''        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )
        def _rank1_verify_mtp(_mod, _inp, _out):
            if not _refusal.is_enabled() or not get_pp_group().is_first_rank:
                return
            if getattr(_mod, "_rank1_writers_verified", False):
                return
            # Separate keys prevent one writer from hiding a missing other one.
            _refusal.verify_consumed("mtp.fc_hidden", 1)
            _refusal.verify_consumed("mtp.fc_embedding", 1)
            _mod._rank1_writers_verified = True

        self.register_forward_hook(_rank1_verify_mtp)
''',1),
   ('models/qwen4_exp/nvidia/ple_layer.py','from vllm.forward_context import get_forward_context\n','from vllm.forward_context import get_forward_context\nfrom vllm import refusal_projection as _refusal\n',1),
   ('models/qwen4_exp/nvidia/ple_layer.py','        self._short_conv(conv_input, gated_output, hidden_states)\n        return gated_output\n',
'''        if _refusal.is_enabled():
            # The nightly kernel includes the outer residual. Project only the
            # PLE writer, without recovering its delta by a lossy subtraction.
            self._short_conv(conv_input, gated_output, torch.zeros_like(hidden_states))
            return hidden_states + _refusal.project_hc(
                gated_output, self.hc_count, key=self.prefix
            )
        self._short_conv(conv_input, gated_output, hidden_states)
        return gated_output
''',1)]
 hashes={path:hashlib.sha256((SRC/path).read_bytes()).hexdigest() for path,*_ in ops}
 if name=='patch_vllm_qwen38next':
  for path in ['models/qwen4_exp/nvidia/ops/ple.py','model_executor/models/registry.py']:
   hashes[path]=hashlib.sha256((SRC/path).read_bytes()).hexdigest()
 (OUT/(name+'.json')).write_text(json.dumps({'commit':'7f1a5398e9610d96c473931a26c0e12bbe0d0423','sha256':hashes,'replacements':ops},indent=2)+'\n')
 (OUT/(name+'.py')).write_text('''#!/usr/bin/env python3
"""Strict refusal port for vLLM nightly 7f1a5398; CPU validated, GPU unvalidated."""
from pathlib import Path
from port_common import main

if __name__ == "__main__":
    main(Path(__file__))
''')
