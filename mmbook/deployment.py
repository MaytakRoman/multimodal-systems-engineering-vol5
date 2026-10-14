"""Глава 10: поканальное INT8, SVD и ограниченная очередь кадров."""
from collections import deque
import numpy as np
from .common import rng, write_json


def quantize_rows(w):
    w=np.asarray(w,float)
    if w.ndim!=2 or not np.isfinite(w).all(): raise ValueError('Нужна конечная матрица')
    maximum=np.max(np.abs(w),axis=1,keepdims=True)
    scale=np.where(maximum>0,maximum/127.,1.)
    q=np.clip(np.rint(w/scale),-127,127).astype(np.int8)
    return q,scale.astype(np.float32)


def low_rank(w, rank):
    if not 0<rank<=min(w.shape): raise ValueError('Недопустимый ранг')
    u,s,v=np.linalg.svd(w,full_matrices=False)
    return (u[:,:rank]*s[:rank])@v[:rank]


def simulate_queue(arrivals, service_ms, capacity):
    """Один worker, FIFO, drop-new при переполнении; времена в миллисекундах."""
    if capacity<1 or service_ms<=0 or any(b<a for a,b in zip(arrivals,arrivals[1:])):
        raise ValueError('Неверная очередь')
    completions=deque(); accepted=[]; dropped=[]
    for i,t in enumerate(arrivals):
        while completions and completions[0]<=t: completions.popleft()
        if len(completions)>=capacity: dropped.append(i); continue
        end=max(t,completions[-1] if completions else t)+service_ms
        completions.append(end); accepted.append(dict(frame=i,arrival=t,completion=end,latency=end-t))
    return dict(accepted=accepted,dropped=dropped)


def experiment(out, seed):
    """Сравнить ошибки сжатия и переполнение очереди на расчётной нагрузке."""
    r=rng(seed); w=(r.normal(size=(48,8))@r.normal(size=(8,64))*.1).astype(np.float32)
    x=r.normal(size=(64,32)).astype(np.float32); q,scale=quantize_rows(w); restored=q.astype(float)*scale
    y=w@x; approx=restored@x; rank8=low_rank(w,8)
    np.save(out/'weights_fp32.npy',w,allow_pickle=False); np.save(out/'weights_int8.npy',q,allow_pickle=False)
    np.save(out/'scales.npy',scale,allow_pickle=False)
    queue=simulate_queue(list(range(0,1000,30)),45,3)
    write_json(out/'queue.json',queue)
    return dict(fp32_payload_bytes=w.nbytes,int8_payload_bytes=q.nbytes+scale.nbytes,
                quantized_output_relative_l2=float(np.linalg.norm(y-approx)/np.linalg.norm(y)),
                rank8_weight_relative_l2=float(np.linalg.norm(w-rank8)/np.linalg.norm(w)),
                queue_frames=len(queue['accepted'])+len(queue['dropped']),dropped=len(queue['dropped']),
                p95_simulated_latency_ms=float(np.percentile([a['latency'] for a in queue['accepted']],95)),
                limitation='Dequantized NumPy matmul, not an INT8 device kernel. Bytes exclude container headers. Queue times are simulated, not measured device latency.')
