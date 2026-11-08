from pathlib import Path
import argparse,ctypes as C,json
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--candidate',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists(): raise SystemExit('Preserve previous operator results')
p=C.POINTER(C.c_float)
functions=[]
for name in [args.reference,args.candidate]:
 lib=C.CDLL(str(name));f=lib.oracle_attention;f.argtypes=[p,p,p,p]+[C.c_int]*6;f.restype=C.c_int;functions.append(f)
rng=np.random.default_rng(45123);report=[]
for batch,start,heads,width,capacity in [(1,0,2,128,1024),(8,13,2,128,1024),(31,609,2,128,1024),(64,12,2,128,1024),(128,35,2,128,1024),(1,639,2,128,1024),(1,1023,2,128,1024)]:
 q=rng.normal(size=(batch,heads,width)).astype('f');k=rng.normal(size=(heads,capacity,width)).astype('f');v=rng.normal(size=k.shape).astype('f');ys=[]
 for f in functions:
  y=np.zeros_like(q);rc=f(*[a.ctypes.data_as(p) for a in [q,k,v,y]],batch,start,heads,width,capacity,8);assert rc==0,rc;ys.append(y)
 d=ys[0].astype('d')-ys[1];r=dict(batch=batch,start=start,heads=heads,width=width,maxAbs=float(abs(d).max()),relativeRms=float(np.linalg.norm(d)/np.linalg.norm(ys[0])),equal=bool(np.array_equal(*ys)));report.append(r);print(json.dumps(r),flush=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')

if not all(r["equal"] for r in report): raise SystemExit("Operator arithmetic differs")
