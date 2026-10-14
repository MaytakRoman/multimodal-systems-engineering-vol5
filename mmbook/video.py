"""Глава 6: оконное покрытие длинного видео, подсчёт пересечений и события."""
import numpy as np
from .common import write_json


def windows(length, size, overlap):
    if length<=0 or size<=0 or not 0<=overlap<size: raise ValueError('Неверные параметры окна')
    result=[]; start=0
    while start<length:
        result.append((start,min(start+size,length)))
        if start+size>=length: break
        start+=size-overlap
    return result


def count_crossings(tracks, line=0.5):
    """Слева направо, один объект не может быть посчитан дважды."""
    counts=[]
    for ident, points in sorted(tracks.items()):
        points=sorted(points)
        for (t0,x0),(t1,x1) in zip(points,points[1:]):
            if x0<line<=x1:
                counts.append(dict(track_id=ident,time=t1)); break
    return counts


def event_intervals(active, min_frames=3):
    if min_frames<=0: raise ValueError('Неверная длительность')
    a=np.asarray(active,bool); intervals=[]; start=None
    for i,value in enumerate(np.r_[a,False]):
        if value and start is None: start=i
        if not value and start is not None:
            if i-start>=min_frames: intervals.append((start,i))
            start=None
    return intervals


def experiment(out, seed):
    """Сопоставить оконный контекст, кадры тревоги и число уникальных событий."""
    tracks={'part-1':[(0,.1),(1,.4),(2,.7),(3,.4),(4,.8)],
            'part-2':[(0,.2),(1,.3),(2,.4)],'part-3':[(0,.9),(1,.6),(2,.2)]}
    signal=[False]*5+[True]*7+[False]*4+[True]+[False]*4+[True]*6
    intervals=event_intervals(signal,3)
    write_json(out/'tracks.json',tracks); write_json(out/'event_signal.json',signal)
    return dict(windows=windows(3600,300,60),unique_crossings=count_crossings(tracks),
                active_frames=sum(signal),events=intervals,event_count=len(intervals),
                limitation='Tracks and activity are fixture inputs, not outputs of a video detector.')
