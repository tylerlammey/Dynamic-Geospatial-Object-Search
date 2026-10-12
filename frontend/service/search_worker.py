"""Frontend-owned one-shot queries or an explicitly loaded persistent GPU model."""
import argparse
import json
import os
import re
from pathlib import Path
import sys

from storage import atomic_json, contained
from retrieval import ExactRetrieval


def run_request(frontend,request_path,adapter,before_complete=lambda:None):
    request_path=contained(frontend,request_path);request=json.loads(request_path.read_text(encoding='utf-8'))
    task=request['task_id']
    if not isinstance(task,str) or len(task)!=32 or any(c not in '0123456789abcdef' for c in task):raise ValueError('Invalid task ID')
    status_path=frontend/'logs/search'/task/'status.json';result_path=frontend/'state/searches'/f'{task}.json'
    def progress(value):atomic_json(status_path,{'task_id':task,**value},frontend)
    try:
        result=adapter.text(request,progress)
        atomic_json(result_path,result,frontend)
        # Bound disposable query history without deleting bookmarks/provenance.
        for old in sorted((frontend/'state/searches').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[30:]:
            contained(frontend,old).unlink()
        for old in sorted((frontend/'state/text-vectors').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[200:]:
            contained(frontend,old).unlink()
        before_complete();progress({'state':'complete','result_id':task})
    except Exception as exc:
        before_complete()
        progress({'state':'error','error':str(exc)})
        return 1
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frontend',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--request');mode.add_argument('--resident',action='store_true')
    parser.add_argument('--allow-partial',action='store_true');parser.add_argument('--online',action='store_true')
    parser.add_argument('--worker-id')
    args=parser.parse_args();frontend=Path(args.frontend).resolve();repository=frontend.parent
    if args.resident and not re.fullmatch('[a-f0-9]{32}',args.worker_id or ''):
        parser.error('--resident requires a valid --worker-id')
    sys.path.insert(0,str(repository/'src'))
    adapter=ExactRetrieval(repository,frontend)
    if not args.resident:return run_request(frontend,Path(args.request),adapter)
    def model_progress(value):
        atomic_json(frontend/'logs/model-status.json',{'pid':os.getpid(),'worker_id':args.worker_id,**value},frontend)
    def ready():
        model_progress({'state':'ready','run_fingerprint':adapter.resident_fingerprint,'gpu':adapter.resident_contract.get('gpu')})
    failed=False
    try:
        adapter.preload(args.allow_partial,not args.online,model_progress)
        for line in sys.stdin:
            command=json.loads(line)
            if command.get('action')=='unload':break
            if command.get('action')!='search':raise ValueError('Invalid model worker command')
            request_path=contained(frontend,frontend/command['request'])
            model_progress({'state':'searching','run_fingerprint':adapter.resident_fingerprint})
            run_request(frontend,request_path,adapter,ready)
        model_progress({'state':'unloading'})
    except Exception as exc:
        failed=True;model_progress({'state':'error','error':str(exc)})
    finally:
        adapter.close()
        if not failed:model_progress({'state':'unloaded'})
    return int(failed)


if __name__=='__main__':raise SystemExit(main())
