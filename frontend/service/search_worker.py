"""Frontend-owned short-lived query process, never an embedding controller."""
import argparse
import json
from pathlib import Path
import sys

from storage import atomic_json, contained
from retrieval import ExactRetrieval


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frontend',required=True);parser.add_argument('--request',required=True)
    args=parser.parse_args();frontend=Path(args.frontend).resolve();repository=frontend.parent
    sys.path.insert(0,str(repository/'src'))
    request_path=contained(frontend,Path(args.request));request=json.loads(request_path.read_text(encoding='utf-8'))
    task=request['task_id']
    if not isinstance(task,str) or len(task)!=32 or any(c not in '0123456789abcdef' for c in task):raise ValueError('Invalid task ID')
    status_path=frontend/'logs/search'/task/'status.json';result_path=frontend/'state/searches'/f'{task}.json'
    def progress(value):atomic_json(status_path,{'task_id':task,**value},frontend)
    try:
        result=ExactRetrieval(repository,frontend).text(request,progress)
        atomic_json(result_path,result,frontend)
        # Bound disposable query history without deleting bookmarks/provenance.
        for old in sorted((frontend/'state/searches').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[30:]:
            contained(frontend,old).unlink()
        for old in sorted((frontend/'state/text-vectors').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[200:]:
            contained(frontend,old).unlink()
        progress({'state':'complete','result_id':task})
    except Exception as exc:
        progress({'state':'error','error':str(exc)})
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
