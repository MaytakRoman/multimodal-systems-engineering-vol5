"""Глава 8: простая модель движения, синусоидальный звук и сетка OBJ."""
import numpy as np
from .common import rng, write_json, write_pgm
from .speech import save_wave


def fit_dynamics(states):
    a=np.asarray(states,float)
    if a.ndim!=2 or len(a)<3: raise ValueError('Нужно не меньше трёх состояний')
    return np.linalg.lstsq(np.c_[a[:-1],np.ones(len(a)-1)],a[1:],rcond=None)[0]


def rollout(initial, weights, steps):
    if steps<0: raise ValueError('Отрицательное число шагов')
    states=[np.asarray(initial,float)]
    for _ in range(steps): states.append(np.r_[states[-1],1.]@weights)
    return np.asarray(states)


def height_mesh(path, z):
    """Сохранить регулярную поверхность OBJ с треугольными гранями."""
    h,w=z.shape
    if min(h,w)<2: raise ValueError('Сетка слишком мала')
    lines=[]
    for i in range(h):
        for j in range(w): lines.append(f'v {j/(w-1):.6f} {i/(h-1):.6f} {z[i,j]:.6f}')
    for i in range(h-1):
        for j in range(w-1):
            a=i*w+j+1; lines.extend((f'f {a} {a+1} {a+w}',f'f {a+1} {a+w+1} {a+w}'))
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(vertices=h*w,triangles=2*(h-1)*(w-1))


def experiment(out, seed):
    """Обучить affine динамику движения; сохранить кадры, WAV и 3D-поверхность."""
    r=rng(seed); t=np.arange(100)
    states=np.c_[.08+.003*t,.2+.004*t]
    train=states[:60]+r.normal(0,.0005,(60,2)); weights=fit_dynamics(train)
    prediction=rollout(states[59],weights,40)
    for i,state in enumerate(prediction[::4]):
        image=np.zeros((64,64)); x,y=np.rint(state*63).astype(int)
        x,y=np.clip([x,y],2,61); image[y-2:y+3,x-2:x+3]=220
        write_pgm(out/f'frame_{i:03}.pgm',image)
    rate=16000; seconds=2; at=np.arange(rate*seconds)/rate
    audio=sum(.12*np.sin(2*np.pi*f*at) for f in (220,275,330))*np.sin(np.pi*at/seconds)**2
    save_wave(out/'synthetic_chord.wav',audio,rate)
    yy,xx=np.mgrid[-1:1:20j,-1:1:20j]; mesh=height_mesh(out/'surface.obj',.15*np.exp(-3*(xx**2+yy**2)))
    write_json(out/'world_rollout.json',dict(weights=weights,truth=states[59:],prediction=prediction))
    return dict(rollout_mse=float(np.mean((prediction-states[59:])**2)),train_frames=60,test_frames=40,
                saved_frames=11,audio_seconds=seconds,mesh=mesh,
                limitation='Affine state model, procedural frames/audio/mesh; no generative video/audio/3D foundation model is run.')
