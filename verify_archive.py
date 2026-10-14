"""Проверить контрольные суммы исходного комплекта без его изменения."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    expected=json.loads((root/'SHA256.json').read_text(encoding='utf-8'))
    errors=[]
    for name,sha in expected.items():
        path=root/name
        if not path.is_file(): errors.append(f'missing: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=sha: errors.append(f'changed: {name}')
    if errors: raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(expected)} files match SHA256.json')


if __name__=='__main__': main()
