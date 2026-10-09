#!/usr/bin/env python3
"""Reads a GGUF tensor table from a local prefix and reports architecture + tensor types."""
import struct, sys, collections
TYPE={0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
      10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',30:'BF16'}
T={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def main(path):
    f=open(path,'rb')
    if f.read(4)!=b'GGUF': print("NOT_GGUF"); return 1
    struct.unpack('<I',f.read(4))
    nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
    def rs():
        n=struct.unpack('<Q',f.read(8))[0]; return f.read(n).decode('utf-8','replace')
    def rv(t):
        if t==8: return rs()
        if t==9:
            et=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
            if et==8:
                for _ in range(n): rs()
                return None
            sz=struct.calcsize(T[et]); f.read(sz*n); return None
        sz=struct.calcsize(T[t]); return struct.unpack('<'+T[t],f.read(sz))[0]
    arch=None; pooling=None
    for _ in range(nkv):
        k=rs(); t=struct.unpack('<I',f.read(4))[0]; v=rv(t)
        if k=='general.architecture': arch=v
        if k.endswith('.pooling_type'): pooling=v
    types=collections.Counter()
    for _ in range(nt):
        rs(); nd=struct.unpack('<I',f.read(4))[0]
        for _ in range(nd): struct.unpack('<Q',f.read(8))[0]
        ty=struct.unpack('<I',f.read(4))[0]; struct.unpack('<Q',f.read(8))
        types[TYPE.get(ty,f'?{ty}')]+=1
    print(f"arch={arch} pooling={pooling} types={dict(types)}")
    return 0
if __name__=='__main__': sys.exit(main(sys.argv[1]))
