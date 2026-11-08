from pathlib import Path
import torch,numpy as np,json,hashlib,sys
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
x=torch.linspace(-8,8,8193,dtype=torch.float32); y=torch.nn.functional.gelu(x,approximate='none');x.numpy().tofile(out/'gelu-input.f32');y.numpy().tofile(out/'gelu-torch.f32')
state=0x12345678
def floats(n):
 global state
 a=np.empty(n,dtype='f')
 for i in range(n):
  state=(1664525*state+1013904223)&0xffffffff;a[i]=np.float32((state>>8)/16777216)*np.float32(.5)-np.float32(.25)
 return a
batch,inputs,outputs=577,1024,16
x=floats(batch*inputs).reshape(batch,inputs);w=floats(outputs*inputs).reshape(outputs,inputs);b=floats(outputs)
y=torch.nn.functional.linear(torch.from_numpy(x),torch.from_numpy(w),torch.from_numpy(b)).numpy();y.tofile(out/'affine-torch.f32')
report=dict(torch=torch.__version__,mkldnn=torch.backends.mkldnn.enabled,affine=dict(batch=batch,inputs=inputs,outputs=outputs,seed=0x12345678,generator='LCG u32: state=1664525*state+1013904223; ((state>>8)/2^24)*0.5-0.25; input then weights then bias'),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.f32')})
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
