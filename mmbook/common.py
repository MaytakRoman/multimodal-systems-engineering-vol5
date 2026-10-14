"""Общие операции, форматы и контракт командной строки."""
import argparse
import json
from pathlib import Path
import numpy as np


def rng(seed):
    return np.random.Generator(np.random.PCG64(seed))


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


def normalize(x):
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Нулевой вектор не имеет косинусного представления")
    return x / norms


def binary_metrics(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    if y.shape != pred.shape or y.size == 0:
        raise ValueError("Нужны непустые массивы одинаковой формы")
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    fn, tn = int((y & ~pred).sum()), int((~y & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,
                accuracy=(tp+tn)/y.size)


def select_threshold(y, scores):
    """Порог F1 на validation; при равенстве выбирается больший порог."""
    scores = np.asarray(scores, float)
    if scores.shape != np.asarray(y).shape or not np.isfinite(scores).all():
        raise ValueError("Недопустимые оценки")
    candidates = np.unique(np.r_[0.0, scores, 1.000001])
    return float(max(candidates, key=lambda t: (binary_metrics(y, scores >= t)['f1'], t)))


def json_ready(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [json_ready(v) for v in x]
    return x


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def write_pgm(path, image):
    """PGM P5: серое изображение без внешнего графического кодека."""
    a = np.clip(image, 0, 255).astype(np.uint8)
    if a.ndim != 2: raise ValueError("Нужен двумерный массив")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(f"P5\n{a.shape[1]} {a.shape[0]}\n255\n".encode()+a.tobytes())


def read_pgm(path):
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'P5': raise ValueError("Нужен PGM P5")
        w, h = map(int, f.readline().split())
        if f.readline().strip() != b'255': raise ValueError("Нужен 8-битный PGM")
        a = np.frombuffer(f.read(), dtype=np.uint8)
    if a.size != w*h: raise ValueError("Неверная длина растра")
    return a.reshape(h, w)


def cli(chapter, experiment):
    p = argparse.ArgumentParser(description=experiment.__doc__)
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    out = a.output / f'chapter{chapter:02}'
    out.mkdir(parents=True, exist_ok=True)
    result = experiment(out, a.seed)
    result.update(chapter=chapter, seed=a.seed, data_origin='synthetic',
                  model_scope='educational numerical implementation; no pretrained weights')
    write_json(out / 'summary.json', result)
    print(json.dumps(json_ready(result), ensure_ascii=False, sort_keys=True))
