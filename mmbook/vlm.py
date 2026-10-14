"""Глава 3: бюджет визуальных токенов и контроль утверждений."""
import numpy as np
from .common import rng, write_json


def token_budget(context, text, tiles, per_tile, reserved):
    if any(x < 0 for x in (context, text, tiles, per_tile, reserved)): raise ValueError('Отрицательный бюджет')
    used = text + tiles*per_tile + reserved
    return dict(used=used, remaining=context-used, fits=used<=context)


def gate_claim(claim, observed_object, observed_attribute, min_confidence=0.8):
    """Утверждение проходит только при совпадении предмета, признака и уверенности."""
    if claim['object_id'] != observed_object['id']: return 'wrong_object'
    if not observed_object['visible']: return 'not_visible'
    if observed_attribute['confidence'] < min_confidence: return 'low_confidence'
    if claim['attribute'] != observed_attribute['value']: return 'unsupported_attribute'
    return 'accepted'


def risk_coverage(correct, confidence, thresholds):
    correct, confidence = np.asarray(correct, bool), np.asarray(confidence, float)
    if correct.shape != confidence.shape or not len(correct): raise ValueError('Несогласованные данные')
    rows = []
    for t in thresholds:
        chosen = confidence >= t; n = int(chosen.sum())
        rows.append(dict(threshold=t, accepted=n, coverage=n/len(chosen),
                         risk=float((~correct[chosen]).mean()) if n else None))
    return rows


def experiment(out, seed):
    """Проверить отказ от непроверенных визуальных утверждений на размеченных записях."""
    r = rng(seed); records=[]
    for i in range(200):
        truth = bool(r.random() > 0.25)
        confidence = float(r.uniform(0.65, 1) if truth else r.uniform(0.1, 0.92))
        records.append(dict(id=i, correct=truth, confidence=confidence))
    write_json(out/'claim_records.json', records)
    obj=dict(id='pump-17', visible=True); attr=dict(value='red', confidence=0.91)
    claims=[dict(object_id='pump-17', attribute='red'), dict(object_id='pump-18', attribute='red'),
            dict(object_id='pump-17', attribute='green')]
    return dict(budget=token_budget(8192, 3100, 8, 256, 1200),
                gate=[dict(claim=c, status=gate_claim(c, obj, attr)) for c in claims],
                risk_coverage=risk_coverage([a['correct'] for a in records], [a['confidence'] for a in records], [0, .5, .75, .9, 1]),
                limitation='No VLM inference or training; observed attributes and confidence are synthetic inputs.')
