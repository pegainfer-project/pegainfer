#!/usr/bin/env python3
"""Exercise the production HIP sources against independent CPU Torch math."""
import ctypes as C
import json
import sys
import numpy as np
import torch

torch.manual_seed(20261112)
torch.set_num_threads(2)
lib=C.CDLL(sys.argv[1]);P=C.c_void_p;I=C.c_int;S=C.c_size_t
specs={
'pgi_create':[I,C.POINTER(P)],'pgi_alloc':[P,S,C.POINTER(P)],'pgi_copy':[P,P,P,S,I],
'pgi_free':[P,P],'pgi_sync':[P],
'pgi_norm':[P,P,P,P,I,I,C.c_float],'pgi_gemm':[P,P,P,P,I,I,I],
'pgi_add':[P,P,P,P,S],'pgi_silu':[P,P,P,P,S],
'pgi_rope':[P,P,P,I,I,I,C.c_float],
'pgi_cache':[P,P,P,P,P,P,P,I,I,I,I,I],
'pgi_attention':[P,P,P,P,P,P,P,I,I,I,I,I,I],
'pgi_top1':[P,P,P,I,I],
}
for name,args in specs.items():getattr(lib,name).argtypes=args;getattr(lib,name).restype=I
lib.pgi_last_error.restype=C.c_char_p
lib.pgi_destroy.argtypes=[P]
def call(name,*args):
 status=getattr(lib,name)(*args)
 if status:raise RuntimeError((name,status,lib.pgi_last_error().decode()))
d=P();call('pgi_create',0,C.byref(d));allocations=[];report=[]
def gpu(t):
 a=t.contiguous().view(torch.uint16).numpy() if t.dtype==torch.bfloat16 else t.contiguous().numpy()
 p=P();call('pgi_alloc',d,a.nbytes,C.byref(p));allocations.append(p)
 call('pgi_copy',d,p,P(a.ctypes.data),a.nbytes,1);return p

def read(p,shape,dtype=torch.bfloat16):
 a=np.zeros(shape,dtype=np.uint16 if dtype==torch.bfloat16 else np.int32)
 call('pgi_copy',d,P(a.ctypes.data),p,a.nbytes,0)
 return torch.from_numpy(a).view(dtype)
def compare(name,actual,expected,atol=0.004,rtol=0.01):
 torch.testing.assert_close(actual.float(),expected.float(),atol=atol,rtol=rtol)
 report.append({'op':name,'max_abs':float((actual.float()-expected.float()).abs().max()),'passed':True})
try:
 x=torch.randn(5,128).to(torch.bfloat16);w=torch.randn(128).to(torch.bfloat16);out=gpu(torch.zeros_like(x))
 call('pgi_norm',d,gpu(x),gpu(w),out,5,128,1e-6)
 compare('rmsnorm',read(out,x.shape),((x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6))*w.float()).bfloat16(),atol=0.02)
 a=torch.randn(7,128).bfloat16();b=torch.randn(65,128).bfloat16();o=gpu(torch.zeros(7,65,dtype=torch.bfloat16))
 call('pgi_gemm',d,gpu(b),gpu(a),o,7,65,128)
 compare('bf16_gemm',read(o,(7,65)),(a.float()@b.float().T).bfloat16(),atol=0.125)
 call('pgi_add',d,gpu(x),gpu(x),out,x.numel());compare('residual',read(out,x.shape),(x+x),atol=0,rtol=0)
 call('pgi_silu',d,gpu(x),gpu(x),out,x.numel());compare('silu_gate',read(out,x.shape),torch.nn.functional.silu(x)*x,atol=0.02)
 positions=torch.tensor([0,1,0,2],dtype=torch.int32);slots=torch.tensor([0,0,1,0],dtype=torch.int32)
 q=torch.randn(4,4,128).bfloat16();k=torch.randn(4,2,128).bfloat16();v=torch.randn_like(k)
 qdev=gpu(q);p=gpu(positions);slot=gpu(slots)
 call('pgi_rope',d,qdev,p,4,4,128,1000000.)
 angle=positions.float()[:,None]*1000000.**(-torch.arange(0,128,2).float()/128)
 c=torch.cat([angle.cos(),angle.cos()],-1).bfloat16()[:,None,:];s=torch.cat([angle.sin(),angle.sin()],-1).bfloat16()[:,None,:]
 rotated=q*c+torch.cat([-q[:,:,64:],q[:,:,:64]],-1)*s
 compare('rope',read(qdev,q.shape),rotated,atol=0.02)
 high_positions=torch.tensor([127,511,4095,8191],dtype=torch.int32)
 high_q=gpu(q)
 call('pgi_rope',d,high_q,gpu(high_positions),4,4,128,1000000.)
 high_angle=high_positions.float()[:,None]*1000000.**(-torch.arange(0,128,2).float()/128)
 high_c=torch.cat([high_angle.cos(),high_angle.cos()],-1).bfloat16()[:,None,:]
 high_s=torch.cat([high_angle.sin(),high_angle.sin()],-1).bfloat16()[:,None,:]
 compare('rope_high_positions',read(high_q,q.shape),q*high_c+torch.cat([-q[:,:,64:],q[:,:,:64]],-1)*high_s,atol=0.02)
 keys=gpu(torch.zeros(2,16,2,128,dtype=torch.bfloat16));values=gpu(torch.zeros(2,16,2,128,dtype=torch.bfloat16))
 call('pgi_cache',d,gpu(k),gpu(v),keys,values,slot,p,4,2,128,16,2)
 attn=gpu(torch.zeros_like(q))
 call('pgi_attention',d,gpu(q),keys,values,attn,slot,p,4,4,2,128,16,2)
 expected=torch.empty_like(q)
 for row in range(4):
  idx=[j for j in range(4) if slots[j]==slots[row] and positions[j]<=positions[row]]
  for head in range(4):
   scores=q[row,head].float()@k[idx,head//2].float().T/(128**.5)
   expected[row,head]=(scores.softmax(-1)@v[idx,head//2].float()).bfloat16()
 compare('ragged_causal_attention',read(attn,q.shape),expected,atol=0.008)
 long_k=torch.randn(1,8192,2,128).bfloat16();long_v=torch.randn_like(long_k)
 long_q=q[:1];long_out=gpu(torch.zeros_like(long_q));long_ref=torch.empty_like(long_q)
 call('pgi_attention',d,gpu(long_q),gpu(long_k),gpu(long_v),long_out,gpu(torch.tensor([0],dtype=torch.int32)),gpu(torch.tensor([8191],dtype=torch.int32)),1,4,2,128,8192,1)
 for head in range(4):
  scores=long_q[0,head].float()@long_k[0,:,head//2].float().T/(128**.5)
  long_ref[0,head]=(scores.softmax(-1)@long_v[0,:,head//2].float()).bfloat16()
 compare('attention_context8192',read(long_out,long_q.shape),long_ref,atol=0.001)
 logits=torch.tensor([[1.,2.,2.,-1.],[-1.,-1.,-2.,-4.]],dtype=torch.bfloat16);ids=gpu(torch.zeros(2,dtype=torch.int32))
 call('pgi_top1',d,gpu(logits),ids,2,4)
 assert read(ids,(2,),torch.int32).tolist()==[1,0]
 report.append({'op':'argmax_tie','passed':True})
 print(json.dumps(report,indent=2))
finally:
 for p in allocations:call('pgi_free',d,p)
 lib.pgi_destroy(d)
