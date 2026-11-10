from pathlib import Path
import ctypes as c, numpy as np, json, hashlib, sys
out=Path(sys.argv[1]);out.mkdir();libdir=Path('/opt/vision/reference/source/build/bin')
base=c.CDLL(str(libdir/'libggml-base.so'),mode=c.RTLD_GLOBAL);ref=c.CDLL(str(libdir/'libggml-cpu.so'))
base.ggml_quantize_chunk.argtypes=[c.c_int,c.c_void_p,c.c_void_p,c.c_int64,c.c_int64,c.c_int64,c.c_void_p];base.ggml_quantize_chunk.restype=c.c_size_t
ref.ggml_cpu_init()
ref.quantize_row_q8_K.argtypes=[c.c_void_p,c.c_void_p,c.c_int64]
ref.ggml_vec_dot_q5_K_q8_K.argtypes=[c.c_int,c.c_void_p,c.c_size_t,c.c_void_p,c.c_size_t,c.c_void_p,c.c_size_t,c.c_int]
libs=[]
for name,path in [('baseline','/opt/vision/final-repeat/native-candidate.so'),('candidate','/opt/vision/q5-native-v2/native-candidate.so')]:
 lib=c.CDLL(path);lib.jmodels_kernels_context_create.argtypes=[c.c_uint32];lib.jmodels_kernels_context_create.restype=c.c_void_p
 lib.jmodels_quantized_f32_batched_matmul_with_context.argtypes=[c.c_void_p,c.c_uint32,c.c_void_p,c.c_uint64,c.c_void_p,c.c_uint64,c.c_void_p,c.c_uint64,c.c_uint32,c.c_uint32,c.c_uint32]
 lib.jmodels_kernels_context_destroy.argtypes=[c.c_void_p]
 libs.append((name,lib,lib.jmodels_kernels_context_create(3)))
state=371
def floats(n):
 global state
 a=np.empty(n,dtype='<f4')
 for i in range(n):
  state=(1664525*state+1013904223)&0xffffffff;a[i]=np.float32(state>>8)*np.float32(1/16777216)*np.float32(8)-np.float32(4)
 return a
report=[]
for cols in [256,768,4096,11008]:
 rows=9;batch=17;weights=floats(cols*rows);input=floats(cols*batch);packed=np.zeros(rows*(cols//256)*176,dtype='u1')
 assert base.ggml_quantize_chunk(13,weights.ctypes.data,packed.ctypes.data,0,rows,cols,None)==packed.nbytes
 q8=np.zeros(batch*(cols//256)*292,dtype='u1');expected=np.empty((batch,rows),dtype='<f4')
 for b in range(batch):
  yp=q8.ctypes.data+b*(cols//256)*292;ref.quantize_row_q8_K(input.ctypes.data+b*cols*4,yp,cols)
  for row in range(rows):ref.ggml_vec_dot_q5_K_q8_K(cols,expected.ctypes.data+(b*rows+row)*4,0,packed.ctypes.data+row*(cols//256)*176,0,yp,0,1)
 folder=out/str(cols);folder.mkdir();packed.tofile(folder/'weights.q5k');input.tofile(folder/'input.f32');expected.tofile(folder/'expected.f32')
 for n in [1,3,8,17]:
  for name,lib,ctx in libs:
   actual=np.empty((n,rows),dtype='<f4');status=lib.jmodels_quantized_f32_batched_matmul_with_context(ctx,4,packed.ctypes.data,packed.nbytes,input.ctypes.data,n*cols,actual.ctypes.data,n*rows,n,rows,cols);assert status==0,status
   actual.tofile(folder/(name+'-'+str(n)+'.f32'));target=expected[:n]
   report.append(dict(arm=name,cols=cols,rows=rows,batch=n,exact=int(np.sum(actual.view('u4')==target.view('u4'))),total=n*rows,maxAbs=float(abs(actual-target).max())))
for _,lib,ctx in libs:lib.jmodels_kernels_context_destroy(ctx)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'hashes.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},indent=2)+'\n')
print(json.dumps(report),flush=True)
assert all(x['exact']==x['total'] for x in report if x['arm']=='candidate')
(out/'COMPLETE').touch()
