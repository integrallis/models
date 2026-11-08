from pathlib import Path
import ctypes as c,numpy as np,json,hashlib
r=Path('/opt/vision/reference-complete');lib=c.CDLL(str(r/'native-candidate.so'))
lib.jmodels_kernels_context_create.argtypes=[c.c_uint32];lib.jmodels_kernels_context_create.restype=c.c_void_p
lib.jmodels_quantized_f32_batched_matmul_with_context.argtypes=[c.c_void_p,c.c_uint32,c.c_void_p,c.c_uint64,c.c_void_p,c.c_uint64,c.c_void_p,c.c_uint64,c.c_uint32,c.c_uint32,c.c_uint32]
lib.jmodels_kernels_context_destroy.argtypes=[c.c_void_p]
rows=[]
for kind,code in [('q5',12),('q6',13)]:
 for cols in [768,4096]:
  d=r/'graph-oracle'/kind/str(cols);w=np.fromfile(d/'weights.bin',dtype='u1');x=np.fromfile(d/'input.f32',dtype='<f4')
  for threads in [1,3,8]:
   ctx=lib.jmodels_kernels_context_create(threads)
   for batch in [1,3,7,8,17]:
    target=np.fromfile(d/f'expected-{batch}.f32',dtype='<f4');actual=np.empty_like(target)
    status=lib.jmodels_quantized_f32_batched_matmul_with_context(ctx,code,w.ctypes.data,w.nbytes,x.ctypes.data,batch*cols,actual.ctypes.data,actual.size,batch,8,cols);assert status==0,status
    rows.append(dict(kind=kind,cols=cols,batch=batch,threads=threads,exact=int(np.sum(actual.view('u4')==target.view('u4'))),total=target.size,maxAbs=float(abs(actual-target).max())))
   lib.jmodels_kernels_context_destroy(ctx)
(r/'graph-oracle/report.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps([x for x in rows if x['exact']!=x['total']]),flush=True);assert all(x['exact']==x['total'] for x in rows);print('ALL 60 SHAPES BIT EXACT',flush=True)
