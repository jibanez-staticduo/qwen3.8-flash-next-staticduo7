"""Execute the actual router transaction AST with an adversarial CPU engine."""
import ast
import asyncio
import logging
import math
from pathlib import Path
from types import SimpleNamespace as NS

class AsyncMPClient: pass
class Response:
 def __init__(self,content,status_code=200):self.content=content;self.status_code=status_code
source=Path(__file__).parents[1].joinpath('payload/refusal_api_router.py').read_text()
keep={'_Control','_control','_all_ranks','_supported_workers','engine_client','_rpc','set_refusal_lambda','get_refusal_lambda'}
nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in keep]
for n in nodes:
 if hasattr(n,'decorator_list'):n.decorator_list=[]
ns={'asyncio':asyncio,'math':math,'AsyncMPClient':AsyncMPClient,'JSONResponse':Response,'Request':object,'EngineClient':object,'RefusalLambdaRequest':object,'logger':logging.getLogger('test')}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_router_transaction','exec'),ns)
logging.disable(logging.CRITICAL)

class Engine:
 def __init__(self):
  self.engine_core=AsyncMPClient();self._client_count=1
  self.vllm_config=NS(parallel_config=NS(data_parallel_size=1,world_size=2))
  self.trace=[];self.paused=False;self.pending=0;self.values=[0.,0.]
  self.reset_ok=True;self.setter=None;self.readback=None;self.pause_gate=None
  self.started=asyncio.Event();self.resume_error=False;self.cache_epoch=0
 async def is_paused(self):return self.paused
 async def pause_generation(self,*,mode,clear_cache):
  assert mode=='wait' and clear_cache is True
  self.trace.append('pause');self.paused=True;self.started.set()
  if self.pause_gate:await self.pause_gate.wait()
  if not self.reset_ok:raise RuntimeError('cache reset failed')
  self.cache_epoch+=1;self.trace.append('reset')
 def get_num_unfinished_requests(self):return self.pending
 async def collective_rpc(self,method,args=()):
  self.trace.append(method)
  if method=='set_refusal_lambda':
   assert self.paused and self.cache_epoch>0
   if isinstance(self.setter,Exception):
    self.values[0]=args[0];raise self.setter
   self.values=[args[0]]*2
   return self.values if self.setter is None else self.setter
  return self.values if self.readback is None else self.readback
 async def resume_generation(self):
  self.trace.append('resume')
  if self.resume_error:raise RuntimeError('resume failed')
  self.paused=False

def request(e):return NS(app=NS(state=NS(engine_client=e)))
async def set_(e,value=1.,raw=None):
 return await ns['set_refusal_lambda'](NS(lambda_=value),raw or request(e))

async def main():
 e=Engine();r=await set_(e)
 assert r.status_code==200 and r.content['cache_reset'] and e.values==[1.,1.] and not e.paused
 assert e.trace==['pause','reset','set_refusal_lambda','get_refusal_lambda','resume']
 print('PASS reset precedes all worker mutations, exact TP2 readback precedes resume')
 e=Engine();e.pause_gate=asyncio.Event();raw=request(e)
 t=asyncio.create_task(set_(e,raw=raw));await e.started.wait()
 assert e.values==[0.,0.] and e.trace==['pause']
 e.pause_gate.set();assert (await t).status_code==200
 print('PASS running-work drain blocks mutation until pause/cache completion')
 for pending in (1,5):
  e=Engine();e.pending=pending;r=await set_(e)
  assert r.status_code==409 and e.values==[0.,0.] and not e.paused
  assert 'set_refusal_lambda' not in e.trace
 print('PASS queued/preempted requests reject transition and resume unchanged')
 e=Engine();e.paused=True;r=await set_(e)
 assert r.status_code==409 and not e.trace and e.paused
 print('PASS pre-existing pause is never acquired or resumed')
 for change in ('inproc','clients','dp'):
  e=Engine()
  if change=='inproc':e.engine_core=object()
  if change=='clients':e._client_count=2
  if change=='dp':e.vllm_config.parallel_config.data_parallel_size=2
  assert (await set_(e)).status_code==503 and not e.trace
 print('PASS inproc, multi-API and DP>1 reject before side effects')
 e=Engine();e.reset_ok=False;raw=request(e);r=await set_(e,raw=raw)
 assert r.status_code==503 and e.paused and e.values==[0.,0.] and 'set_refusal_lambda' not in e.trace
 n=len(e.trace);assert (await set_(e,raw=raw)).status_code==503 and len(e.trace)==n
 print('PASS failed reset never mutates; uncertain pause blocks subsequent setters')
 for result in ([1.,None],[1.],[1.,0.],[1.,float('nan')],[],[True,1.],RuntimeError('partial RPC')):
  e=Engine();e.setter=result;r=await set_(e)
  assert r.status_code==503 and e.paused and 'resume' not in e.trace
 print('PASS missing, None, mismatched, NaN, bool and partial-error worker replies stay paused')
 e=Engine();e.readback=[1.,0.];r=await set_(e)
 assert r.status_code==503 and e.paused and 'resume' not in e.trace
 e=Engine();e.resume_error=True;r=await set_(e)
 assert r.status_code==503 and e.paused
 print('PASS readback/resume failures never report success')
 e=Engine();e.pause_gate=asyncio.Event();raw=request(e)
 first=asyncio.create_task(set_(e,1.,raw));await e.started.wait()
 second=asyncio.create_task(set_(e,0.,raw));await asyncio.sleep(0)
 assert e.trace==['pause']
 e.pause_gate.set();a,b=await asyncio.gather(first,second)
 assert a.status_code==b.status_code==200 and e.values==[0.,0.]
 assert e.trace==['pause','reset','set_refusal_lambda','get_refusal_lambda','resume']*2
 print('PASS concurrent setters serialize complete transactions')
 e=Engine();e.pause_gate=asyncio.Event();raw=request(e)
 t=asyncio.create_task(set_(e,raw=raw));await e.started.wait();t.cancel()
 try:await t
 except asyncio.CancelledError:pass
 else:raise AssertionError('cancellation swallowed')
 assert e.paused and ns['_control'](raw).blocked and 'resume' not in e.trace
 print('PASS cancellation leaves uncertain pause blocked')
 for value in (float('nan'),float('inf')):
  e=Engine();assert (await set_(e,value)).status_code==422 and not e.trace
 print('PASS nonfinite requests reject before side effects')

asyncio.run(main())
