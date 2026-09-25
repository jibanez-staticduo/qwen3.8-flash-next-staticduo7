#!/usr/bin/env python3
import ast
import argparse
import hashlib
import os
from pathlib import Path
EXPECTED_SHA256 = '1a1cd94fb44e3ef6a6ae44103b12a1891ba7aedecc4f1d983ac090cafb27eaa6'
HELPER = r'''

def _attach_draft_vocab(model: nn.Module) -> None:
    path = os.environ.get('VLLM_MTP_DRAFT_VOCAB', '').strip()
    if not path:
        return
    lm_head = getattr(model, 'lm_head', None)
    weight = getattr(lm_head, 'weight', None)
    shard = getattr(lm_head, 'shard_indices', None)
    if weight is None or weight.dim() != 2 or shard is None:
        return
    if model.logits_processor.scale <= 0:
        raise ValueError('Reduced draft vocabulary requires positive logit scale')
    try:
        with open(path) as handle:
            ids = sorted({int(line) for line in handle if line.strip()})
    except OSError as exc:
        logger.warning('MTP draft vocab: cannot read %s (%s); skipping', path, exc)
        return
    org_vocab = int(getattr(lm_head, 'org_vocab_size', weight.shape[0]))
    ids = [i for i in ids if 0 <= i < org_vocab]
    if not ids or len(ids) >= org_vocab:
        raise ValueError('Reduced draft vocabulary is empty or not reduced')
    start, end = int(shard.org_vocab_start_index), int(shard.org_vocab_end_index)
    local_ids = [i for i in ids if start <= i < end]
    rows = torch.tensor([i - start for i in local_ids], dtype=torch.long, device=weight.device)
    model.register_buffer('_draft_lm_head_weight', weight.data.index_select(0, rows).contiguous(), persistent=False)
    model.register_buffer('_draft_id_to_target_id', torch.tensor(local_ids, dtype=torch.long, device=weight.device), persistent=False)
    logger.info('MTP draft vocab: %d of %d ids, %d local', len(ids), org_vocab, len(local_ids))
'''
METHOD = r'''

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = getattr(self, '_draft_lm_head_weight', None)
        if weight is None:
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
        if weight.shape[0] == 0:
            values = torch.full((hidden_states.shape[0],), float('-inf'), device=hidden_states.device)
            ids = torch.zeros_like(values, dtype=torch.long)
        else:
            logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)
            values, local = logits.max(dim=-1)
            ids = self._draft_id_to_target_id[local]
        if getattr(self.lm_head, 'tp_size', 1) == 1:
            return ids.to(torch.int64)
        pair = torch.stack([values.float(), ids.float()], dim=-1)
        gathered = tensor_model_parallel_all_gather(pair, dim=-1).view(hidden_states.shape[0], self.lm_head.tp_size, 2)
        winner = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(dim=-1, index=winner).squeeze(-1).to(torch.int64)
'''
def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f'Expected one anchor, found {count}: {old[:80]!r}')
    return source.replace(old, new, 1)


def build(source: Path, output: Path) -> None:
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError(f'Unsupported Qwen4Exp MTP source: {actual}')
    s = raw.decode('utf-8')
    s = replace_once(s, 'import regex as re\n', 'import os\nimport regex as re\n')
    s = replace_once(s, 'from vllm.distributed import get_pp_group\n',
                     'from vllm.distributed import get_pp_group\nfrom vllm.distributed.communication_op import tensor_model_parallel_all_gather\n')
    s = replace_once(s, 'def _remap_ignored_layers(\n',
                     "logger = __import__('vllm.logger', fromlist=['init_logger']).init_logger(__name__)\n"
                     + HELPER + '\ndef _remap_ignored_layers(\n')
    s = replace_once(s, '        return self.logits_processor(self.lm_head, hidden_states)\n\n    def load_weights',
                     '        return self.logits_processor(self.lm_head, hidden_states)\n' + METHOD + '\n    def load_weights')
    s = replace_once(s, '        return loader.load_weights(remap_weight_names(), mapper=mapper)\n',
                     '        loaded = loader.load_weights(remap_weight_names(), mapper=mapper)\n'
                     '        _attach_draft_vocab(self)\n        return loaded\n')
    ast.parse(s)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(s, encoding='utf-8')
    print(f'patched {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
