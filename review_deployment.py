"""Контроль SVD матрицей полного ранга, без измерений ускорителя."""
from pathlib import Path
import json,numpy as np

def run(out,seed=2027):
 r=np.random.default_rng(seed);w=r.normal(size=(48,64)).astype(np.float32);x=r.normal(size=(64,32)).astype(np.float32);u,s,v=np.linalg.svd(w,full_matrices=False);rows=[]
 for k in [4,8,16,32,48]:
  a=(u[:,:k]*s[:k])@v[:k];rows.append(dict(rank=k,weight_relative_frobenius=float(np.linalg.norm(w-a)/np.linalg.norm(w)),output_relative_frobenius=float(np.linalg.norm(w@x-a@x)/np.linalg.norm(w@x)),two_factor_fp32_bytes=k*(48+64)*4))
 out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(seed=seed,shape=[48,64],rank_numeric=int(np.linalg.matrix_rank(w)),rows=rows),indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',default='review_results/full_rank_svd.json');a=p.parse_args();run(a.output)
