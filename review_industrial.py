"""Таблица по партиям, кластерные интервалы и трёхканальный синтетический контроль."""
import json,re
from pathlib import Path
import numpy as np
from mmbook.common import binary_metrics

def run(source,out,seed=2026):
 source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 data=json.loads((source/'predictions.json').read_text());test=[r for r in data if r['split']=='test']
 lots=sorted({r['lot'] for r in test});bylot=[];counts=[]
 for lot in lots:
  rows=[r for r in test if r['lot']==lot];m=binary_metrics(np.array([r['target'] for r in rows]),np.array([r['prediction'] for r in rows]));bylot.append(dict(lot=lot,n=len(rows),**m));counts.append([m[k] for k in ['tp','fp','fn','tn']])
 random=np.random.default_rng(seed);weights=random.multinomial(len(lots),[1/len(lots)]*len(lots),size=20000);totals=weights@np.array(counts);tp,fp,fn,tn=totals.T
 bs={'f1':2*tp/(2*tp+fp+fn),'precision':tp/(tp+fp),'recall':tp/(tp+fn),'accuracy':(tp+tn)/(tp+fp+fn+tn)}
 ci={k:[float(v) for v in np.percentile(a,[2.5,97.5],method='linear')] for k,a in bs.items()}
 # Each object has three separate inputs: image score, textual route and equipment log.
 r=np.random.default_rng(seed+1);joined=[]
 for i,row in enumerate(test):
  ident=row['id'];lo,hi=[(8.,12.),(10.,14.),(12.,16.)][i%3]
  route=f'Операция: затяжка; станция: S1; момент: [{lo:.1f}; {hi:.1f}] Н·м'
  bad=bool(r.random()<.18);torque=(hi+1.5) if bad else (lo+hi)/2
  event=dict(part_id=ident,operation='затяжка',station='S1',torque_nm=torque,age_ms=150)
  missing=i%17==0;wrong=i%29==3
  if missing:event=None
  elif wrong:event['part_id']=ident+'-other'
  match=re.fullmatch(r'Операция: (.+); станция: (.+); момент: \[([0-9.]+); ([0-9.]+)\] Н·м',route);assert match
  operation,station,low,high=match.groups();valid=event is not None and event['part_id']==ident and event['operation']==operation and event['station']==station and event['age_ms']<=500
  target=int(bool(row['target']) or bad)
  status='review' if not valid else ('reject' if row['prediction'] or not float(low)<=event['torque_nm']<=float(high) else 'accept')
  joined.append(dict(id=ident,lot=row['lot'],image=row['image'],image_score=row['score'],image_prediction=row['prediction'],route=route,event=event,target_joint=target,status=status,valid_join=valid))
 covered=[x for x in joined if x['valid_join']];y=np.array([x['target_joint'] for x in covered]);base=np.array([x['image_prediction'] for x in covered]);fusion=np.array([x['status']=='reject' for x in covered]);summary=dict(seed=seed,scope='synthetic images, controlled text grammar and simulated equipment events; no field deployment',n=len(test),covered=len(covered),coverage=len(covered)/len(test),review=len(test)-len(covered),image_only_on_same_covered=binary_metrics(y,base),fusion_on_covered=binary_metrics(y,fusion),visual_test_by_lot=bylot,visual_test_cluster_bootstrap=dict(repeats=20000,seed=seed,groups=len(lots),method='percentile, 2.5/97.5%, linear quantile, fixed fitted model and validation threshold',ci95=ci))
 # Controlled sensor perturbation: true target and joining coverage remain fixed.
 noise_seed=seed+2;noise_rng=np.random.default_rng(noise_seed)
 z=noise_rng.standard_normal(len(covered));noise_rows=[];noise_inputs=[]
 for sigma in (0.,0.5,1.,2.):
  predictions=[];values=[]
  for j,x in enumerate(covered):
   match=re.fullmatch(r'Операция: (.+); станция: (.+); момент: \[([0-9.]+); ([0-9.]+)\] Н·м',x['route']);low,high=map(float,match.groups()[2:])
   true_torque=x['event']['torque_nm'];measured=true_torque+sigma*z[j]
   pred=int(bool(x['image_prediction']) or not low<=measured<=high)
   predictions.append(pred);values.append(dict(id=x['id'],target=x['target_joint'],true_torque_nm=true_torque,measured_torque_nm=float(measured),noise_standard_normal=float(z[j]),prediction=pred))
  noise_rows.append(dict(sigma_nm=sigma,**binary_metrics(y,np.array(predictions))))
  noise_inputs.append(dict(sigma_nm=sigma,objects=values))
 # Monte Carlo sensor uncertainty on the fixed 131 objects (not a sample CI).
 repetitions=1000;draws=np.random.default_rng(noise_seed).standard_normal((repetitions,len(covered)))
 true=[];low=[];high=[]
 for x in covered:
  match=re.fullmatch(r'Операция: (.+); станция: (.+); момент: \[([0-9.]+); ([0-9.]+)\] Н·м',x['route']);l,h=map(float,match.groups()[2:]);low.append(l);high.append(h);true.append(x['event']['torque_nm'])
 true=np.array(true);low=np.array(low);high=np.array(high);monte_carlo=[]
 for sigma in (0.,0.5,1.,2.):
  measured=true[None,:]+sigma*draws
  decisions=base[None,:].astype(bool)|(measured<low[None,:])|(measured>high[None,:])
  tp=(decisions&(y[None,:]==1)).sum(1);fp=(decisions&(y[None,:]==0)).sum(1);fn=((~decisions)&(y[None,:]==1)).sum(1);f1=2*tp/(2*tp+fp+fn)
  monte_carlo.append(dict(sigma_nm=sigma,mean_fp=float(fp.mean()),mean_fn=float(fn.mean()),mean_f1=float(f1.mean()),f1_percentile_2_5_97_5=[float(v) for v in np.percentile(f1,[2.5,97.5])]))
 summary['sensor_noise']=dict(seed=noise_seed,distribution='additive Gaussian N(0,sigma^2), independent across objects; shared standard normal draws across sigma levels',n=len(covered),target='unchanged true scratch OR true torque violation',rows=noise_rows,monte_carlo_repetitions=repetitions,monte_carlo=monte_carlo)
 (out/'sensor_noise_inputs_and_decisions.json').write_text(json.dumps(noise_inputs,ensure_ascii=False,indent=2))
 (out/'assembly_inputs_and_decisions.json').write_text(json.dumps(joined,ensure_ascii=False,indent=2));(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--source',default='results/chapter11');p.add_argument('--output',default='review_results/industrial');a=p.parse_args();run(a.source,a.output)
