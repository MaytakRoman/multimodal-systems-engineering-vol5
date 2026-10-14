"""Подготовить ссылки для рукописи по фактическому адресу опубликованного репозитория."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit, quote


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,help='https://github.com/OWNER/REPOSITORY')
    p.add_argument('--ref',default='main',help='Ветка или SHA коммита')
    p.add_argument('--output',type=Path,default=Path('.'))
    args=p.parse_args(); url=args.repo.rstrip('/')
    parsed=urlsplit(url); parts=parsed.path.strip('/').split('/')
    if parsed.scheme!='https' or parsed.netloc!='github.com' or len(parts)!=2 or not all(parts) or parsed.query or parsed.fragment:
        p.error('Нужен корневой HTTPS-адрес GitHub-репозитория без /tree/main')
    url=url.removesuffix('.git')
    root=Path(__file__).resolve().parent
    rows=json.loads((root/'book_links.json').read_text(encoding='utf-8'))
    for row in rows: row['url']=url+'/blob/'+quote(args.ref,safe='')+'/'+row['relative_path']
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'links_for_manuscript.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text='# Ссылки для рукописи\n\n'+'\n'.join(f"- {r['section']} {r['title']}: {r['url']}" for r in rows)+'\n'
    (args.output/'links_for_manuscript.md').write_text(text,encoding='utf-8')
    print('Created links_for_manuscript.json and links_for_manuscript.md')


if __name__=='__main__': main()
