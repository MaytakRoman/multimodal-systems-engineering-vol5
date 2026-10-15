"""Глава 9: наблюдение-изображение, неоднозначность и обязательный проход."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from mmbook.robotics import shortest_path,simulate

def run(out,seed=2026):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 grid=np.zeros((12,12),int);grid[:,5]=1;grid[7,5]=0
 image=np.full((12,12,3),255,np.uint8);image[grid==1]=(0,0,0)
 image[1,1]=(0,0,255);image[10,10]=(255,0,0);image[2,9]=(255,0,0)
 Image.fromarray(np.repeat(np.repeat(image,20,0),20,1)).save(out/'observation.png')
 # The policy receives the image, not a separate hidden object table.
 observed=np.array(Image.open(out/'observation.png'))[10::20,10::20]
 recovered=np.all(observed==[0,0,0],axis=-1).astype(int)
 red=np.argwhere(np.all(observed==[255,0,0],axis=-1));start=tuple(map(int,np.argwhere(np.all(observed==[0,0,255],axis=-1))[0]))
 def resolve(command):
  if command=='go to the red box':return dict(status='needs_clarification',candidates=red.tolist(),actions=0)
  if command=='go to the lower red box':return dict(status='resolved',goal=red[np.argmax(red[:,0])].tolist())
  return dict(status='unsupported_command',actions=0)
 ambiguous=resolve('go to the red box');resolved=resolve('go to the lower red box');goal=tuple(map(int,resolved['goal']))
 path=shortest_path(recovered,start,goal);assert (7,5) in path and len(path)-1==18
 sealed=recovered.copy();sealed[7,5]=1;assert shortest_path(sealed,start,goal) is None
 r=np.random.default_rng(seed);rows=[]
 for slip in [0,.1,.3,.5]:
  eps=[simulate(recovered,start,goal,r,slip) for _ in range(60)]
  rows.append(dict(slip=slip,episodes=60,successes=sum(x['success'] for x in eps),mean_steps=float(np.mean([x['steps'] for x in eps])),collisions=sum(x['collisions'] for x in eps)))
  (out/f'episodes_{slip}.json').write_text(json.dumps(eps,indent=2))
 summary=dict(seed=seed,start=list(map(int,start)),goal=list(map(int,goal)),path=[list(map(int,p)) for p in path],ambiguous=ambiguous,resolved=resolved,transfer=rows,wall='column 5, rows 0..11 except row 7',vision='exact synthetic RGB palette',language='two explicitly supported command forms',scope='symbolic sensor and parser, not a learned VLA or physical robot')
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',default='review_results/robotics');p.add_argument('--seed',type=int,default=2026);a=p.parse_args();run(a.output,a.seed)
