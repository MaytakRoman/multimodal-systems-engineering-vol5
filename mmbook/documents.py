"""Глава 4: восстановление таблицы из OCR-боксов, нормализация и проверки документа."""
from decimal import Decimal
import re
from .common import write_json


def group_rows(tokens, tolerance=5):
    """Кластеризация центров OCR-боксов по Y; координаты в пикселях одного листа."""
    if tolerance < 0: raise ValueError('Отрицательный допуск')
    rows=[]
    for token in sorted(tokens, key=lambda a: (a['y'], a['x'])):
        match = next((row for row in rows if abs(sum(t['y'] for t in row)/len(row)-token['y'])<=tolerance), None)
        if match is None: rows.append([token])
        else: match.append(token)
    return [sorted(row, key=lambda a:a['x']) for row in rows]


def parse_decimal(text):
    value = text.replace('\u00a0','').replace(' ','').replace(',','.')
    if not re.fullmatch(r'[+-]?\d+(\.\d+)?', value): raise ValueError('Неверное число')
    return Decimal(value)


def validate_invoice(rows, stated_total, tolerance=Decimal('0.01')):
    if tolerance < 0: raise ValueError('Отрицательный допуск')
    calculated = sum((parse_decimal(row['qty'])*parse_decimal(row['price']) for row in rows), Decimal(0))
    delta = abs(calculated-parse_decimal(stated_total))
    return dict(calculated_total=str(calculated), delta=str(delta), accepted=delta<=tolerance)


def drawing_status(nominal, measured, lower, upper, unit):
    if lower > upper or unit not in ('mm','cm'): raise ValueError('Неверный контракт размера')
    factor = 10 if unit=='cm' else 1
    deviation=measured*factor-nominal
    return dict(deviation_mm=deviation, accepted=lower<=deviation<=upper)


def experiment(out, seed):
    """Восстановить строки OCR и обнаружить несогласованные суммы и размеры."""
    tokens=[dict(text='Наименование', x=10, y=20), dict(text='Количество', x=160, y=21),
            dict(text='Цена', x=260, y=19), dict(text='Болт', x=10, y=60),
            dict(text='4', x=160, y=62), dict(text='12,50', x=260, y=60),
            dict(text='Гайка', x=10, y=99), dict(text='2', x=160, y=101), dict(text='8,00', x=260, y=100)]
    table=[[t['text'] for t in row] for row in group_rows(tokens)]
    write_json(out/'ocr_tokens.json', tokens); write_json(out/'reconstructed_table.json', table)
    rows=[dict(qty=r[1],price=r[2]) for r in table[1:]]
    return dict(table=table, valid_total=validate_invoice(rows,'66,00'),
                wrong_total=validate_invoice(rows,'68,00'),
                technical_dimension=drawing_status(20,2.003,-.05,.05,'cm'),
                mismatched_unit=drawing_status(20,2.003,-.05,.05,'mm'),
                limitation='OCR boxes are fixture inputs, not recognition output. No OCR or formula model is loaded.')
