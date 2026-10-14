"""Глава 2: патчи, маски, координаты и доменная калибровка."""
import numpy as np
from .common import rng, binary_metrics, select_threshold, write_json, write_pgm


def patchify(image, patch):
    a = np.asarray(image)
    if a.ndim != 2 or patch <= 0 or any(s % patch for s in a.shape):
        raise ValueError('Размер изображения должен делиться на размер патча')
    h, w = a.shape
    return a.reshape(h//patch, patch, w//patch, patch).transpose(0, 2, 1, 3).reshape(-1, patch*patch)


def restore_box(box, crop_origin, scale):
    """XYXY из масштабированного crop в исходную систему; scale = resized/original."""
    b = np.asarray(box, float)
    if b.shape != (4,) or scale <= 0 or b[2] < b[0] or b[3] < b[1]: raise ValueError('Неверная рамка')
    return b/scale + np.tile(crop_origin, 2)


def mask_metrics(reference, prediction):
    a, b = np.asarray(reference, bool), np.asarray(prediction, bool)
    if a.shape != b.shape: raise ValueError('Размеры масок различаются')
    inter, union = (a & b).sum(), (a | b).sum()
    return dict(iou=float(inter/union) if union else 1.0,
                dice=float(2*inter/(a.sum()+b.sum())) if a.sum()+b.sum() else 1.0)


def experiment(out, seed):
    """Изучить маски, token budget и перенос порога на синтетический домен."""
    r = rng(seed)
    image = r.normal(90, 6, (64, 64)); mask = np.zeros((64, 64), bool)
    mask[22:39, 27:43] = True; image[mask] += 80
    pred = image > 130
    write_pgm(out/'image.pgm', image); write_pgm(out/'reference_mask.pgm', mask*255)
    write_pgm(out/'prediction_mask.pgm', pred*255)
    y = r.integers(0, 2, 400)
    scores = np.clip(0.12+0.30*y+r.normal(0, 0.10, 400), 0, 1)
    threshold = select_threshold(y[:160], scores[:160])
    write_json(out/'domain_predictions.json', [dict(id=i, split='validation' if i<160 else 'test',
               target=int(y[i]), score=scores[i]) for i in range(400)])
    token_rows = [dict(resolution=s, patch=16, tokens=(s//16)**2,
                       attention_pairs=(s//16)**4) for s in (224, 448, 896)]
    return dict(patches=patchify(image, 16).shape, segmentation=mask_metrics(mask, pred),
                restored_box=restore_box([10, 20, 50, 60], [100, 200], 0.5),
                validation_threshold=threshold,
                test_default=binary_metrics(y[160:], scores[160:]>=0.5),
                test_calibrated=binary_metrics(y[160:], scores[160:]>=threshold),
                token_budget=token_rows,
                limitation='Text-conditioned detector and ViT weights are not executed; mask is intensity thresholding.')
