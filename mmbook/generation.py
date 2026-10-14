"""Глава 7: орнштейн-уленбековская диффузия, flow и дистилляция в 2D."""
import numpy as np
from .common import rng, write_json, write_pgm


def diffuse(x, alpha_bar, noise):
    if not 0<=alpha_bar<=1: raise ValueError('alpha_bar вне [0,1]')
    if np.shape(x)!=np.shape(noise): raise ValueError('Различные формы')
    return np.sqrt(alpha_bar)*x+np.sqrt(1-alpha_bar)*noise


def flow_euler(z, target, steps, stiffness=2.0):
    """dx/dt = stiffness*(target-x); аналитический endpoint используется только для проверки."""
    if steps<=0 or stiffness<=0: raise ValueError('Неверные параметры интегрирования')
    x=np.array(z,float,copy=True)
    for _ in range(steps): x+=(stiffness/steps)*(target-x)
    return x


def distill_linear(z_train, teacher_train):
    """Один affine шаг, обученный на endpoint учителя методом наименьших квадратов."""
    z_train, teacher_train=np.asarray(z_train),np.asarray(teacher_train)
    if z_train.ndim!=2 or teacher_train.shape!=z_train.shape: raise ValueError('Неверная обучающая пара')
    design=np.c_[z_train,np.ones(len(z_train))]
    return np.linalg.lstsq(design,teacher_train,rcond=None)[0]


def experiment(out, seed):
    """Сравнить шаги интегратора и обучить одношагового ученика в двумерной задаче."""
    r=rng(seed); target=np.array([2.,-1.]); train=r.normal(size=(256,2)); test=r.normal(size=(128,2))
    teacher_train=flow_euler(train,target,64); teacher_test=flow_euler(test,target,64)
    weights=distill_linear(train,teacher_train); student=np.c_[test,np.ones(len(test))]@weights
    exact=target+(test-target)*np.exp(-2.)
    rows=[dict(steps=n,mse_to_exact=float(np.mean((flow_euler(test,target,n)-exact)**2))) for n in [1,2,4,8,16,64]]
    write_json(out/'trajectories.json',dict(initial=test[:8],teacher=teacher_test[:8],student=student[:8],weights=weights))
    yy,xx=np.mgrid[-1:1:64j,-1:1:64j]; image=np.exp(-8*(xx**2+yy**2))*255
    noise=r.normal(size=image.shape)
    for alpha in (.95,.5,.05): write_pgm(out/f'noisy_{alpha}.pgm',(diffuse(image/255,alpha,noise)+1)*100)
    return dict(integration=rows,student_mse_to_teacher=float(np.mean((student-teacher_test)**2)),
                train_samples=len(train),test_samples=len(test),
                limitation='Analytic 2D vector field; this is not a trained image diffusion or flow-matching network. No image-quality metric is claimed.')
