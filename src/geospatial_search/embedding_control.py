"""Lightweight graceful pause and exclusive access to an embedding collection."""
import argparse
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import time

from .download_nys_naip import job_lock
from .paths import EMBEDDINGS, JOB, contained


def worker_active(directory):
    if not (directory/'worker.lock').exists():
        return False
    try:
        with job_lock(directory):
            return False
    except RuntimeError:
        return True


@contextmanager
def idle_collection(directory):
    with ExitStack() as stack:
        try:
            stack.enter_context(job_lock(directory))
        except RuntimeError as exc:
            raise RuntimeError('Collection is busy. Pause embedding with scripts/embed-nys.ps1 -Pause and wait for it to finish before searching or resuming.') from exc
        yield


@contextmanager
def gpu_access(job_root):
    """Do not load two large GPU models, even for different collections."""
    directory=job_root/'embeddings'/'gpu-runtime'
    directory.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        try:
            stack.enter_context(job_lock(directory))
        except RuntimeError as exc:
            raise RuntimeError('Another NYS embedding or search process is using the GPU. Pause that embedding run or wait for its search to finish.') from exc
        yield


def request_pause(directory: Path, timeout=300, poll_interval=.5):
    """Wait for the OS lock to be released, not just a possibly stale status."""
    if not directory.is_dir() or not (directory/'worker.lock').exists():
        raise FileNotFoundError(f'No embedding process or collection at {directory}')
    if worker_active(directory):
        # Zero-byte control signals do not consume the image/data byte allowance.
        (directory/'PAUSE').touch(exist_ok=True)
        print('Pause requested. Finishing and saving the current source tile; waiting for GPU release...',flush=True)
        deadline=time.monotonic()+timeout
        last_message=0
        while worker_active(directory):
            if time.monotonic()>=deadline:
                raise TimeoutError('Pause request is still pending; the worker has not stopped. Do not start search yet. Check -Status or run -Pause again.')
            if time.monotonic()-last_message>=10:
                print('Waiting for the embedding/search process to release the collection...',flush=True)
                last_message=time.monotonic()
            time.sleep(poll_interval)
    status_path=directory/'status.json'
    status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    print(f"Process stopped ({status.get('state','idle')}); {status.get('durable_embedded_chips',0):,} saved embeddings are available for search.",flush=True)
    return status


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=EMBEDDINGS)
    parser.add_argument('--timeout',type=float,default=300)
    parser.add_argument('--check-running',action='store_true',help='Exit 0 if busy, 1 if idle; does not load a model')
    args=parser.parse_args(argv)
    if args.timeout<=0:
        raise ValueError('Pause timeout must be positive')
    directory=contained(JOB/'embeddings',args.output_dir)
    if args.check_running:
        return 0 if worker_active(directory) else 1
    request_pause(directory,args.timeout)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
