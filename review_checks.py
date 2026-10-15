"""Checks of saved review experiment invariants."""
from pathlib import Path
import json,hashlib
R=Path(__file__).parent/'review_results'
def load(p):return json.loads((R/p).read_text())
i=load('industrial/summary.json');assert (i['n'],i['covered'],i['review'])==(144,131,13)
a,b=i['image_only_on_same_covered'],i['fusion_on_covered']
for x in (a,b):assert sum(x[k] for k in ('tp','fp','fn','tn'))==131
assert (a['fn'],b['fn'])==(25,7)
for key,total in [('tp',37),('fp',28),('fn',11),('tn',68)]:assert sum(x[key] for x in i['visual_test_by_lot'])==total
lo,hi=i['visual_test_cluster_bootstrap']['ci95']['f1'];assert lo<74/113<hi
r=load('robotics/summary.json');assert [7,5] in r['path']
s=load('full_rank_svd.json');assert s['rank_numeric']==48
rows=s['rows'];assert all(x['weight_relative_frobenius']>y['weight_relative_frobenius'] for x,y in zip(rows,rows[1:]))
assert rows[3]['two_factor_fp32_bytes']>48*64*4
c=load('open_models/clip_retrieval.json');assert c['top1']==[0,1,2]
noise=i['sensor_noise'];assert noise['seed']==2028 and noise['monte_carlo_repetitions']==1000
assert abs(noise['monte_carlo'][0]['mean_f1']-b['f1'])<1e-12
assert noise['monte_carlo'][-1]['mean_fp']>noise['monte_carlo'][0]['mean_fp']
assert noise['monte_carlo'][-1]['mean_f1']<noise['monte_carlo'][0]['mean_f1']
assert load('industrial/lot_conditions.json')['verified_images']==432
print('Review experiment invariants: passed')
