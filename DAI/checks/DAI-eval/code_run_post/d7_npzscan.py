import numpy as np, sys
from pathlib import Path
from multiprocessing import Pool
R=Path("/home/snics/MER-DA/results")
def scan(p):
    z=np.load(p); nf=0; bad8=0; nfl=0; n8=0
    for k in z.files:
        a=z[k]
        if a.dtype.kind=="f":
            nfl+=a.size; nf+=int((~np.isfinite(a)).sum())
        elif a.dtype==np.int8 and not k.endswith("_pl"):
            n8+=a.size; bad8+=int(((a<0)|(a>3)).sum())
    return p.name,nf,bad8,nfl,n8
if __name__=="__main__":
    fs=sorted((R/"dai1/eval/units").glob("*.npz"))+sorted((R/"dai2/eval/units").glob("*.npz"))
    with Pool(6) as pool: res=pool.map(scan,fs,chunksize=4)
    print("files",len(res),"nonfinite total",sum(r[1] for r in res),"int8 out of 0..3",sum(r[2] for r in res),"float values",sum(r[3] for r in res),"int8 values",sum(r[4] for r in res))
    print([r for r in res if r[1] or r[2]][:10])
