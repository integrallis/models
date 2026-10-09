import struct,sys,collections
TYPE={0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
      10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',30:'BF16'}
BYTES_PER={'F32':(1,4),'F16':(1,2),'BF16':(1,2),'Q4_0':(32,18),'Q5_0':(32,22),'Q5_1':(32,24),
           'Q8_0':(32,34),'Q4_K':(256,144),'Q5_K':(256,176),'Q6_K':(256,210),'Q3_K':(256,110),'Q2_K':(256,84)}
f=open(sys.argv[1],'rb'); f.read(4); struct.unpack('<I',f.read(4))
nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
def rs():
    n=struct.unpack('<Q',f.read(8))[0]; return f.read(n).decode('utf-8','replace')
T={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rv(t):
    if t==8: return rs()
    if t==9:
        et=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
        if et==8: return [rs() for _ in range(n)]
        sz=struct.calcsize(T[et]); return [struct.unpack('<'+T[et],f.read(sz))[0] for _ in range(n)]
    sz=struct.calcsize(T[t]); return struct.unpack('<'+T[t],f.read(sz))[0]
for _ in range(nkv):
    rs(); t=struct.unpack('<I',f.read(4))[0]; rv(t)
by_count=collections.Counter(); by_elems=collections.Counter(); names=collections.defaultdict(list)
for _ in range(nt):
    n=rs(); nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for _ in range(nd)]
    ty=TYPE.get(struct.unpack('<I',f.read(4))[0],'?'); struct.unpack('<Q',f.read(8))
    elems=1
    for d in dims: elems*=d
    by_count[ty]+=1; by_elems[ty]+=elems; names[ty].append(n)
tot=sum(by_elems.values())
print(f"{sys.argv[1]}  tensors={nt}")
for ty,e in by_elems.most_common():
    print(f"  {ty:6s} tensors={by_count[ty]:4d}  weights={e:12,d}  {100.0*e/tot:6.2f}% of weights")
if 'Q5_1' in names:
    print("  Q5_1 tensors:", ", ".join(names['Q5_1'][:8]), "..." if len(names['Q5_1'])>8 else "")
