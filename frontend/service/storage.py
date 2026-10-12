"""Frontend-only atomic writes and a byte-bounded disk LRU, including staging."""
from collections import OrderedDict
import os
from pathlib import Path
import re
import threading
import time
import uuid
from contextlib import contextmanager


def contained(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'Path must stay inside {root}')
    return path


def retry_windows_file_operation(operation):
    """Wait briefly for readers/AV to release a Windows file handle.

    Only sharing/access lock errors are retried, for at most about five
    seconds. Genuine persistent permission failures still reach the caller.
    """
    for attempt in range(30):
        try:return operation()
        except OSError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==29:raise
            time.sleep(min(.01*2**min(attempt,5),.2))


class DiskCache:
    def __init__(self, root, limit):
        self.root = Path(root).absolute()
        if self.root.is_symlink(): raise ValueError('Cache root cannot be a symlink')
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self.limit = int(limit)
        if self.limit < 1: raise ValueError('Cache limit must be positive')
        self.lock = threading.RLock()
        self.entries = OrderedDict()
        self.generation = 0
        for p in sorted(self.root.rglob('*'), key=lambda p:p.stat().st_mtime_ns):
            if p.is_symlink(): raise ValueError('Cache cannot contain symlinks')
            if p.is_file():
                contained(self.root,p)
                self.entries[p] = p.stat().st_size
        self.used = sum(self.entries.values())
        self._evict(0)

    def _path(self, key):
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', key) or '..' in key.split('/'):
            raise ValueError('Invalid cache key')
        return contained(self.root,self.root/key)

    def _evict(self, reserve):
        while self.used + reserve > self.limit and self.entries:
            p,size = self.entries.popitem(last=False)
            contained(self.root,p).unlink(missing_ok=True)
            self.used -= size
        if self.used + reserve > self.limit: raise ValueError('Artifact exceeds frontend cache limit')

    def read(self, key):
        with self.lock:
            p=self._path(key)
            if p not in self.entries: return None
            try: value=p.read_bytes()
            except FileNotFoundError:
                self.used-=self.entries.pop(p);return None
            self.entries.move_to_end(p)
            return value

    def write(self, key, blob, generation=None):
        with self.lock:
            if generation is not None and generation!=self.generation:return
            if len(blob)>self.limit:raise ValueError('Artifact exceeds frontend cache limit')
            p=self._path(key)
            # Eviction happens before staging, so simultaneous old/new files count.
            self._evict(len(blob))
            p.parent.mkdir(parents=True,exist_ok=True)
            temporary=contained(self.root,p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp'))
            try:
                with temporary.open('xb') as f:
                    f.write(blob);f.flush();os.fsync(f.fileno())
                retry_windows_file_operation(lambda:os.replace(temporary,p))
                # Keep the previous entry/accounting until publication succeeds.
                old=self.entries.pop(p,0)
                self.used+=len(blob)-old
                self.entries[p]=len(blob)
            finally:
                retry_windows_file_operation(lambda:temporary.unlink(missing_ok=True))

    def clear(self):
        with self.lock:
            self.generation+=1
            for p in list(self.entries):contained(self.root,p).unlink(missing_ok=True)
            self.entries.clear();self.used=0

    def stats(self):
        with self.lock:return {'bytes':self.used,'limit_bytes':self.limit,'files':len(self.entries)}

    def exists(self,key):
        with self.lock:return self._path(key) in self.entries

    @contextmanager
    def file(self,key):
        # Hold the cache lock through the short raster read so clear/eviction
        # cannot delete a derived TIFF while GDAL owns its handle on Windows.
        with self.lock:
            p=self._path(key)
            if p not in self.entries:yield None;return
            self.entries.move_to_end(p)
            yield p


def atomic_json(path, value, root):
    import json
    p=contained(root,path);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=contained(root,p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp'))
    blob=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
    if len(blob)>64_000_000:raise ValueError('JSON artifact is too large')
    try:
        with temporary.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
        retry_windows_file_operation(lambda:os.replace(temporary,p))
    finally:retry_windows_file_operation(lambda:temporary.unlink(missing_ok=True))


@contextmanager
def frontend_ownership(frontend):
    """One service owns the shared cache; this lock is entirely frontend-local."""
    folder=contained(frontend,Path(frontend)/'logs');folder.mkdir(parents=True,exist_ok=True)
    handle=(folder/'frontend-service.lock').open('a+b')
    try:
        handle.seek(0)
        if handle.seek(0,2)==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise RuntimeError('A frontend service is already open. Close its desktop window or browser development service before starting another.') from exc
        yield
    finally:handle.close()
