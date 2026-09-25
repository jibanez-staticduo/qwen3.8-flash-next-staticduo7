import ast
from pathlib import Path
import types
import torch
import refusal_projection as R

root=Path(__file__).parent
source=(root/'speculator.py').read_text()
cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='AutoRegressiveSpeculator')
fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='propose')
blocks=[]
for hook in ('on_prefill_begin','on_multi_step_decode_begin'):
 pos=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr==hook)
 block=fn.body[pos-1]
 assert isinstance(block,ast.If)
 assert ast.unparse(block.test)=='refusal_projection.is_enabled()'
 blocks.append(compile(ast.Module(body=[block],type_ignores=[]),'patched_draft_phase','exec'))
R.init_from_env(2560,device=torch.device('cpu'))
R.set_lambda(.25)
R.add_request(7,1.)
R.add_request(2,0.)
input_batch=types.SimpleNamespace(idx_mapping_np=[7,2,5],num_scheduled_tokens=[3,2,4],num_tokens=9)
ns={'refusal_projection':R,'self':types.SimpleNamespace(pcp_manager=None),'input_batch':input_batch,'num_reqs':3,'dummy_run':False}
exec(blocks[0],ns)
assert R._STATE.tok[:9].tolist()==[1.,1.,1.,0.,0.,.25,.25,.25,.25]
assert torch.all(R._STATE.tok[9:]==.25)
print('PASS actual draft prefill block retains mixed target row ranges and neutral padding')
exec(blocks[1],ns)
assert R._STATE.tok[:3].tolist()==[1.,0.,.25]
assert torch.all(R._STATE.tok[3:]==.25)
print('PASS actual draft decode block compacts to one lambda per request; FULL/fused fill outside replay')
for value in (0.,1.):
 R.remove_request(7);R.remove_request(2);R.set_lambda(value)
 for block in blocks:
  exec(block,ns)
  assert torch.all(R._STATE.tok==value)
print('PASS draft global lambda0/1 without overrides')
R.add_request(7,1.)
ns['dummy_run']=True
R.set_lambda(0.)
for block in blocks:
 exec(block,ns)
 assert torch.all(R._STATE.tok==0.)
print('PASS dummy runs ignore live request slots')
ns['self'].pcp_manager=object()
try:exec(blocks[0],ns)
except RuntimeError as e:assert 'PCP' in str(e)
else:raise AssertionError('unsupported PCP silently accepted')
print('PASS PCP refuses unsupported layout')
