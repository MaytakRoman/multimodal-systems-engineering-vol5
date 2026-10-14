"""Общий запуск в отдельных процессах, проверка связности и воспроизводимости."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import numpy as np
from mmbook.common import write_json, read_pgm
from mmbook.industrial import features, predict

ROOT=Path(__file__).resolve().parent
MODULES=[
    'chapter01_fusion.py','chapter02_vision.py','chapter03_vlm.py','chapter04_documents.py',
    'chapter05_speech.py','chapter06_video.py','chapter07_generation.py','chapter08_media.py',
    'chapter09_robotics.py','chapter10_deployment.py','reference_multimodal.py',
]


def hashes(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob('*')) if p.is_file()}


def execute(script,output,seed,log_path=None):
    command=[sys.executable,str(ROOT/script)]
    if script!='tests.py': command+=['--output',str(output)]
    if script in MODULES: command+=['--seed',str(seed)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    process=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if log_path: log_path.write_text(process.stdout,encoding='utf-8')
    if process.returncode:
        print(process.stdout)
        raise RuntimeError(f'{script}: return code {process.returncode}')
    return dict(script=script,returncode=process.returncode)


def verify_consistency(output):
    summaries=[json.loads((output/f'chapter{i:02}'/'summary.json').read_text(encoding='utf-8')) for i in range(1,12)]
    assert [s['chapter'] for s in summaries]==list(range(1,12))
    predictions=json.loads((output/'chapter11/predictions.json').read_text(encoding='utf-8'))
    summary=summaries[-1]
    groups={s:{a['lot'] for a in predictions if a['split']==s} for s in ['train','validation','test']}
    assert not groups['train']&groups['test'] and not groups['validation']&groups['test'] and not groups['train']&groups['validation']
    for s in groups:
        assert len([a for a in predictions if a['split']==s])==summary['split_sizes'][s]
    test=[a for a in predictions if a['split']=='test']
    m=summary['test']
    assert sum(a['target']==a['prediction'] for a in test)==m['tp']+m['tn']
    assert all((output/'chapter11'/a['image']).is_file() for a in predictions)
    assert sum(m[k] for k in ['tp','fp','fn','tn'])==len(test)
    saved_model=json.loads((output/'chapter11/model.json').read_text(encoding='utf-8'))
    saved_x=np.array([features(read_pgm(output/'chapter11'/a['image'])) for a in predictions])
    replay=predict(saved_model,saved_x)
    np.testing.assert_allclose(replay,[a['score'] for a in predictions],rtol=0,atol=1e-12)
    assert np.array_equal(replay>=saved_model['threshold'],[a['prediction'] for a in predictions])
    train_x=saved_x[[a['split']=='train' for a in predictions]]
    np.testing.assert_allclose(train_x.mean(axis=0),saved_model['mean'],rtol=0,atol=1e-12)
    return dict(chapters=11,disjoint_lots=True,files_and_predictions_consistent=True,
                saved_model_replays_all_predictions=True,normalization_uses_train_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--skip-repeat',action='store_true',help='Пропустить второй запуск для сравнения байтов')
    args=parser.parse_args()
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    # Рабочий output может содержать старый отчёт; хешируется только результат данного набора.
    logs=ROOT/'execution_logs' if output==ROOT/'results' else output/'execution_logs'
    logs.mkdir(parents=True,exist_ok=True)
    actions=[]
    for script in MODULES+['numerical_checks.py','tests.py']:
        actions.append(execute(script,output,args.seed,logs/(Path(script).stem+'.txt')))
        print(f'OK {script}',flush=True)
    consistency=verify_consistency(output)
    relevant={}
    for i in range(1,12):
        relevant.update({f'chapter{i:02}/{k}':v for k,v in hashes(output/f'chapter{i:02}').items()})
    relevant['numerical_checks.json']=hashlib.sha256((output/'numerical_checks.json').read_bytes()).hexdigest()
    repeat=None
    if not args.skip_repeat:
        with tempfile.TemporaryDirectory() as folder:
            second=Path(folder)
            for script in MODULES+['numerical_checks.py']: execute(script,second,args.seed)
            second_hashes=hashes(second)
            changed=[k for k,v in relevant.items() if second_hashes.get(k)!=v]
            if changed: raise RuntimeError(f'Non-reproducible files: {changed}')
            repeat=dict(status='passed',exactly_equal_files=len(relevant))
    report=dict(version='1.0.0',python=platform.python_version(),numpy=np.__version__,
                platform=platform.platform(),seed=args.seed,commands=actions,
                consistency=consistency,repeat=repeat,result_sha256=relevant)
    report_path=ROOT/'execution_report.json' if output==ROOT/'results' else output/'execution_report.json'
    write_json(report_path,report)
    print(f'PASS: 11 chapters; consistency verified; repeat={repeat}',flush=True)


if __name__=='__main__': main()
