"""Глава 9: команда и наблюдение в сеточном симуляторе, шум переноса."""
from collections import deque
import numpy as np
from .common import rng, write_json


def shortest_path(grid, start, goal):
    """BFS по наблюдаемой карте, 0 свободно, 1 препятствие."""
    h,w=grid.shape
    def free(p): return 0<=p[0]<h and 0<=p[1]<w and not grid[p]
    if not free(start) or not free(goal): return None
    queue=deque([start]); previous={start:None}
    while queue:
        p=queue.popleft()
        if p==goal:
            path=[]
            while p is not None: path.append(p); p=previous[p]
            return path[::-1]
        for dy,dx in [(0,1),(1,0),(0,-1),(-1,0)]:
            n=(p[0]+dy,p[1]+dx)
            if free(n) and n not in previous: previous[n]=p; queue.append(n)
    return None


def simulate(grid, start, goal, random, slip=0., limit=100):
    """Планирование с перепланированием; slip меняет движение на случайное соседнее."""
    if not 0<=slip<=1: raise ValueError('slip вне [0,1]')
    position=start; history=[start]; collisions=0
    for _ in range(limit):
        path=shortest_path(grid,position,goal)
        if path is None: return dict(success=False,history=history,collisions=collisions,reason='unreachable')
        if position==goal: break
        desired=path[1]
        if random.random()<slip:
            dy,dx=[(0,1),(1,0),(0,-1),(-1,0)][int(random.integers(4))]
            desired=(position[0]+dy,position[1]+dx)
        if 0<=desired[0]<len(grid) and 0<=desired[1]<grid.shape[1] and not grid[desired]: position=desired
        else: collisions+=1
        history.append(position)
    return dict(success=position==goal,history=history,collisions=collisions,steps=len(history)-1)


def experiment(out, seed):
    """Оценить перенос политики при шуме действий на независимых траекториях."""
    grid=np.zeros((12,12),int); grid[3:10,5]=1; grid[7,5]=0
    objects={'red_box':(10,10),'blue_box':(2,9)}
    command='bring red_box'; goal=objects[command.split()[-1]]
    rows=[]; r=rng(seed)
    for slip in (0,.1,.3,.5):
        episodes=[simulate(grid,(1,1),goal,r,slip) for _ in range(60)]
        rows.append(dict(slip=slip,episodes=60,successes=sum(e['success'] for e in episodes),
                         mean_steps=float(np.mean([e['steps'] for e in episodes])),
                         collisions=sum(e['collisions'] for e in episodes)))
        write_json(out/f'episodes_{slip}.json',episodes)
    write_json(out/'scene.json',dict(grid=grid,objects=objects,command=command))
    return dict(transfer=rows,policy='BFS with replanning',
                limitation='Symbolic parser and perfect grid observation; not a neural VLA, physical robot or real-world transfer result.')
