"""Глава 1: слияние при пропусках, контрастивная функция потерь и cross-attention."""
import numpy as np
from .common import rng, normalize, softmax, binary_metrics, write_json


def late_fusion(scores, present, weights):
    scores, present, weights = np.asarray(scores), np.asarray(present, bool), np.asarray(weights)
    if scores.shape != present.shape or scores.ndim != 2 or weights.shape != (scores.shape[1],):
        raise ValueError('Несогласованные формы')
    if np.any(weights < 0) or not np.isfinite(scores).all(): raise ValueError('Недопустимые значения')
    effective = present * weights
    denom = effective.sum(axis=1)
    if np.any(denom == 0): raise ValueError('Нет доступной модальности с положительным весом')
    return (scores * effective).sum(axis=1) / denom


def contrastive_loss(image, text, temperature=0.1, positive_mask=None):
    """Симметричная InfoNCE; маска может задавать несколько положительных пар."""
    if temperature <= 0: raise ValueError('temperature должна быть положительной')
    image, text = normalize(image), normalize(text)
    if image.shape != text.shape or image.ndim != 2: raise ValueError('Нужны парные матрицы')
    logits = image @ text.T / temperature
    n = len(image)
    mask = np.eye(n, dtype=bool) if positive_mask is None else np.asarray(positive_mask, bool)
    if mask.shape != (n, n) or not mask.any(axis=0).all() or not mask.any(axis=1).all():
        raise ValueError('Каждый объект должен иметь положительную пару')
    # Средняя вероятность положительного множества, а не искусственных отрицательных дублей.
    row = (softmax(logits) * mask).sum(axis=1)
    col = (softmax(logits.T) * mask.T).sum(axis=1)
    return float(-0.5*(np.log(np.maximum(row, 1e-300)).mean()+np.log(np.maximum(col, 1e-300)).mean()))


def cross_attention(q, k, v):
    q, k, v = map(np.asarray, (q, k, v))
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or q.shape[1] != k.shape[1] or len(k) != len(v):
        raise ValueError('Несогласованные формы Q K V')
    weights = softmax(q @ k.T / np.sqrt(q.shape[1]))
    return weights @ v, weights


def experiment(out, seed):
    """Сравнить обычное среднее с нормированным слиянием при отсутствии звука."""
    r = rng(seed)
    y = r.integers(0, 2, 240)
    scores = np.clip(0.2+0.6*y[:, None]+r.normal(0, 0.20, (240, 3)), 0, 1)
    present = r.random((240, 3)) > 0.3
    present[:, 0] = True
    corrupted = np.where(present, scores, 0)
    fused = late_fusion(corrupted, present, [0.5, 0.3, 0.2])
    visual = r.normal(size=(24, 12)); text = visual+r.normal(0, 0.2, visual.shape)
    aligned, attention = cross_attention(text[:3], visual, visual)
    records = [dict(id=i, target=int(y[i]), scores=scores[i], present=present[i], fused=fused[i]) for i in range(len(y))]
    write_json(out/'predictions.json', records)
    write_json(out/'attention.json', dict(weights=attention, output=aligned))
    return dict(masked_fusion=binary_metrics(y, fused >= 0.5),
                naive_zero_fill=binary_metrics(y, corrupted.mean(axis=1) >= 0.5),
                aligned_loss=contrastive_loss(visual, text),
                shuffled_loss=contrastive_loss(visual, np.roll(text, 1, axis=0)),
                examples=240)
