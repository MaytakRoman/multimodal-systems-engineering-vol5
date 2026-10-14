"""Глава 11: синтетическая дефектоскопия, разбиение по партиям, обучение и внедрение."""
import numpy as np
from .common import rng, write_pgm, read_pgm, write_json, binary_metrics, select_threshold


def features(image):
    """Три объяснимых признака: разброс, контраст хвоста, максимальный градиент."""
    a=np.asarray(image,float)/255
    if a.ndim!=2 or min(a.shape)<2: raise ValueError('Нужно серое изображение не меньше 2x2')
    return np.array([a.std(),np.median(a)-np.percentile(a,2),
                     max(np.abs(np.diff(a,axis=0)).max(),np.abs(np.diff(a,axis=1)).max())])


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-40,40)))


def train_logistic(x, y, steps=1200, lr=.05, penalty=.02):
    """Стандартизация и параметры учатся исключительно на train."""
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=2 or len(x)!=len(y) or not len(y) or set(y)-{0,1}: raise ValueError('Неверные обучающие данные')
    mean=x.mean(axis=0); std=np.maximum(x.std(axis=0),1e-8)
    a=np.c_[(x-mean)/std,np.ones(len(x))]; w=np.zeros(a.shape[1])
    for _ in range(steps):
        gradient=a.T@(sigmoid(a@w)-y)/len(y)
        gradient[:-1]+=penalty*w[:-1]
        w-=lr*gradient
    return dict(mean=mean,std=std,weights=w)


def predict(model,x):
    a=(np.asarray(x)-np.asarray(model['mean']))/np.asarray(model['std'])
    return sigmoid(np.c_[a,np.ones(len(a))]@np.asarray(model['weights']))


def generate_dataset(out,seed):
    """36 партий по 12 изделий; партийная освещённость, шум и контраст дефекта."""
    r=rng(seed); records=[]; xs=[]
    for group in range(36):
        split='train' if group<18 else ('validation' if group<24 else 'test')
        brightness=r.uniform(110,175); noise=r.uniform(3,16)
        for i in range(12):
            target=i%3==0
            image=r.normal(brightness,noise,(32,32))
            mask=np.zeros((32,32),bool)
            if target:
                x=int(r.integers(4,27)); y=int(r.integers(4,22))
                mask[y:y+8,x:x+2]=True
                image[mask]-=r.uniform(22,95)
            # Бездефектные изображения могут иметь тёмную полосу освещения.
            if i%7==0: image[:,0:5]-=r.uniform(5,15)
            image=np.clip(image,0,255).astype(np.uint8)
            ident=f'lot{group:02}_part{i:02}'; relative=f'images/{ident}.pgm'
            write_pgm(out/relative,image); write_pgm(out/f'masks/{ident}.pgm',mask*255)
            # Считать признаки по сохранённым байтам, а не по float до квантования.
            xs.append(features(read_pgm(out/relative)))
            records.append(dict(id=ident,lot=group,split=split,target=int(target),image=relative,
                                mask=f'masks/{ident}.pgm'))
    return records,np.asarray(xs)


def experiment(out,seed):
    """Полный локальный опыт: данные → train → validation threshold → test → журнал."""
    records,x=generate_dataset(out,seed); y=np.array([r['target'] for r in records])
    splits=np.array([r['split'] for r in records]); train=splits=='train'; val=splits=='validation'; test=splits=='test'
    groups={s:{r['lot'] for r in records if r['split']==s} for s in ['train','validation','test']}
    if any(groups[a]&groups[b] for a,b in [('train','test'),('train','validation'),('validation','test')]):
        raise RuntimeError('Утечка партий между выборками')
    model=train_logistic(x[train],y[train]); scores=predict(model,x)
    threshold=select_threshold(y[val],scores[val])
    predictions=[dict(**r,score=float(scores[i]),prediction=int(scores[i]>=threshold)) for i,r in enumerate(records)]
    write_json(out/'dataset.json',records); write_json(out/'predictions.json',predictions)
    write_json(out/'model.json',dict(**model,threshold=threshold,feature_names=['std','tail_contrast','max_gradient']))
    # Геометрия/СИЗ заданы во входном fixture; правила не распознают человека на изображении.
    persons=[dict(id='person-1',inside_hazard_zone=True,helmet=True),
             dict(id='person-2',inside_hazard_zone=True,helmet=False),
             dict(id='person-3',inside_hazard_zone=False,helmet=False)]
    safety_events=[p['id'] for p in persons if p['inside_hazard_zone'] and not p['helmet']]
    write_json(out/'safety_rule.json',dict(input=persons,events=safety_events,origin='synthetic fixture'))
    return dict(samples=len(records),split_sizes={s:int((splits==s).sum()) for s in groups},
                split_lots={s:sorted(g) for s,g in groups.items()},threshold=threshold,
                validation=binary_metrics(y[val],scores[val]>=threshold),
                test=binary_metrics(y[test],scores[test]>=threshold),
                safety_events=safety_events,
                limitation='Synthetic grayscale scratches, hand-crafted features and logistic regression. No real factory, neural defect detector, PPE recognizer or economic effect is measured.')
