"""Pin and execute the nightly engine's pause/reset failure propagation."""
import ast
from concurrent.futures import Future
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, MethodType
from typing import Any, Literal, get_args

src=Path(sys.argv[1])
manifest=json.loads(Path(__file__).with_name('global-dial-contract-hashes.json').read_text())
for rel,sha in manifest.items():
 assert hashlib.sha256((src/rel).read_bytes()).hexdigest()==sha,rel

tree=ast.parse((src/'v1/engine/core.py').read_text())
def method(cls,name):
 c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
 n=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name)
 n.decorator_list=[]
 return n
nodes=[method('EngineCore','_reset_caches'),method('EngineCore','_finish_pause'),method('EngineCoreProc','pause_scheduler')]
ns={'Future':Future,'partial':partial,'Any':Any,'get_args':get_args,'PauseMode':Literal['abort','wait','keep'],'PauseState':NS(PAUSED_ALL='all',PAUSED_NEW='new')}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_core_pause','exec'),ns)
for ok in (False,True):
 calls=[]
 e=NS(_idle_state_callbacks=[],_pause_complete=lambda:False,scheduler=NS(set_pause_state=lambda v:calls.append(('pause_state',v))))
 e.model_executor=NS(collective_rpc=lambda v:calls.append(v))
 def reset(**kw):
  assert kw=={'reset_running_requests':True,'reset_connector':True}
  calls.append('reset');return ok
 e.reset_prefix_cache=reset
 e.reset_mm_cache=lambda:calls.append('mm')
 e.reset_encoder_cache=lambda:calls.append('encoder')
 for n in ('_reset_caches','_finish_pause'):setattr(e,n,MethodType(ns[n],e))
 future=ns['pause_scheduler'](e,mode='wait',clear_cache=True)
 assert isinstance(future,Future) and not future.done()
 assert calls==[('pause_state','new')]
 # Engine invokes this callback only once its stepping/drain loop reaches idle.
 e._idle_state_callbacks[0](e)
 if ok:
  assert future.result() is None
  assert calls==[('pause_state','new'),'synchronize_device','reset','mm','encoder']
 else:
  try:future.result()
  except RuntimeError as exc:assert 'reset' in str(exc)
  else:raise AssertionError('reset failure did not propagate')
  assert calls==[('pause_state','new'),'synchronize_device','reset']
print('PASS exact nightly MP wait pauses new work; future waits for idle; sync/reset run before completion; false reset propagates')
