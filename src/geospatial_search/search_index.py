"""Optional CPU IVF-PQ candidate search with original-vector exact rescoring."""
from __future__ import annotations

from functools import lru_cache
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np

from .download_nys_naip import sha256
from .paths import contained

EFFORTS={'fast':64,'balanced':512,'thorough':4096}
CANDIDATE_FLOORS={'fast':4096,'balanced':131072,'thorough':524288}
FULL_READ_FRACTION=1/128
MAX_SEQUENTIAL_READ_BYTES=128*1024**2


class IndexDependencyUnavailable(RuntimeError):
    pass


def snapshot_signature(collection):
    payload={name:collection.get(name) for name in ('run_fingerprint','embedding_dim','chip_count','shards')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_stat(path):
    value=path.stat()
    return {'bytes':value.st_size,'mtime_ns':value.st_mtime_ns}


def validate_options(backend='auto',effort='balanced'):
    if not isinstance(backend,str) or backend not in ('auto','exact','ann'):
        raise ValueError('Search backend must be auto, exact, or ann')
    if not isinstance(effort,str) or effort not in EFFORTS:
        raise ValueError('Search effort must be fast, balanced, or thorough')


def validate_query(query,dimension):
    query=np.asarray(query,dtype=np.float32)
    if query.shape!=(dimension,) or not np.isfinite(query).all() or not np.isclose(np.linalg.norm(query),1,atol=.002):
        raise ValueError('Require a finite normalized query with the index dimension')
    return query


def validate_exclusions(collection,exclude_ids):
    limits={e['shard_id']:e['chip_count'] for e in collection['shards']}
    excluded=set()
    for pair in exclude_ids or ():
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:
            raise ValueError('Excluded IDs must be (shard_id, embedding_id) pairs')
        shard,row=pair
        if not isinstance(shard,str) or shard not in limits or type(row) is not int or not 0<=row<limits[shard]:
            raise ValueError('Excluded chip ID is outside the collection snapshot')
        excluded.add((shard,row))
    if len(excluded)>20_000:
        raise ValueError('At most 20,000 already-seen IDs may be excluded per request')
    return excluded


def read_manifest(directory,collection):
    path=directory/'search-index'/'manifest.json'
    if not path.exists():
        raise FileNotFoundError('No search index; run scripts/build-search-index.ps1')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('format_version')!=1 or data.get('algorithm')!='ivf-pq':
        raise ValueError('Unsupported search index format')
    if data.get('snapshot_signature')!=snapshot_signature(collection):
        raise ValueError('Search index snapshot differs from the current published collection')
    if not re.fullmatch(r'index-[a-f0-9]{20}\.faiss',data.get('index_file','')):
        raise ValueError('Invalid search index filename')
    if data.get('ntotal')!=collection['chip_count'] or data.get('dimension')!=collection['embedding_dim']:
        raise ValueError('Search index count/dimension differs from collection')
    expected=0
    if len(data.get('sources',[]))!=len(collection['shards']):
        raise ValueError('Search index source mapping differs')
    for source,entry in zip(data['sources'],collection['shards']):
        if (source['start']!=expected or source['shard_id']!=entry['shard_id']
            or source['count']!=entry['chip_count'] or source['source_uri']!=entry['source_uri']):
            raise ValueError('Search index source mapping differs')
        expected+=source['count']
    return path,data


def describe_index(directory,collection):
    try:
        _,data=read_manifest(Path(directory),collection)
        return {'ready':True,'algorithm':'ivf-pq','vectors':data['ntotal'],
                'index_bytes':data['index_bytes'],'default_effort':'balanced'}
    except (FileNotFoundError,ValueError,KeyError,json.JSONDecodeError) as exc:
        return {'ready':False,'algorithm':'exact','reason':str(exc)}


class IndexReader:
    def __init__(self,directory,manifest):
        try:
            import faiss
            from threadpoolctl import ThreadpoolController
        except (ImportError,OSError) as exc:
            raise IndexDependencyUnavailable('CPU Faiss could not load; install requirements-search.txt or use exact search') from exc
        self.directory=directory;self.manifest=manifest
        self.threadpool_controller=ThreadpoolController
        self.sources=manifest['sources']
        self.bundle_paths=[contained(directory/'shards',directory/'shards'/s['shard_id']) for s in self.sources]
        self.starts=np.asarray([s['start'] for s in self.sources],dtype=np.int64)
        self.count=manifest['ntotal'];self.dimension=manifest['dimension']
        path=contained(directory/'search-index',directory/'search-index'/manifest['index_file'])
        if path.stat().st_size!=manifest['index_bytes'] or sha256(path)!=manifest['index_sha256']:
            raise ValueError('Search index checksum or size differs; rebuild the index')
        # Only deserialize a checksummed, locally produced artifact.
        self.index=faiss.read_index(str(path))
        if self.index.ntotal!=self.count or self.index.d!=self.dimension or self.index.metric_type!=faiss.METRIC_INNER_PRODUCT:
            raise ValueError('Native index layout differs from manifest')
        faiss.omp_set_num_threads(min(8,manifest['parameters'].get('threads',8)))
        self.lock=threading.RLock()

    def verify_sources(self):
        for source,root in zip(self.sources,self.bundle_paths):
            for name,stat in source['files'].items():
                if name not in ('embeddings.f16.npy','chips.geoparquet','run.json'):
                    raise ValueError('Invalid index bundle file mapping')
                if file_stat(root/name)!=stat:
                    raise ValueError('An indexed embedding bundle changed; verify it before rebuilding')

    def _score_source(self,source_index,local_ids,query,top_k,excluded):
        source=self.sources[source_index]
        local_ids=np.asarray([i for i in local_ids if (source['shard_id'],int(i)) not in excluded],dtype=np.int64)
        if not len(local_ids):return [],0,False
        # Dense requests use one bounded sequential read rather than many
        # Windows mapping page faults. Sparse requests keep mapped row reads.
        sequential=(len(local_ids)>=source['count']*FULL_READ_FRACTION
            and source['count']*self.dimension*2<=MAX_SEQUENTIAL_READ_BYTES)
        array=np.load(self.bundle_paths[source_index]/'embeddings.f16.npy',
            mmap_mode=None if sequential else 'r',allow_pickle=False)
        try:
            if array.shape!=(source['count'],self.dimension) or array.dtype!=np.float16:
                raise ValueError('Indexed vector layout changed')
            values=array[local_ids].astype(np.float32)
            if not np.isfinite(values).all():raise ValueError('Nonfinite image embeddings')
            scores=values@query
            selected=np.lexsort((local_ids,-scores))[:min(top_k,len(local_ids))]
            rows=[(source['shard_id'],int(local_ids[i]),float(scores[i])) for i in selected]
            return rows,len(local_ids),sequential
        finally:del array

    def search(self,query,top_k,effort,excluded,diagnostics):
        query=validate_query(query,self.dimension)
        available=self.count-len(excluded)
        if available<=0:
            diagnostics.update(backend='ivf-pq',approximate=True,effort=effort,
                candidate_count=0,rescored_count=0,excluded_count=len(excluded),
                index_vectors=self.count,exhausted=True,index_signature=self.manifest['snapshot_signature'])
            return []
        wanted=min(top_k,available)
        candidates=min(self.count,max(CANDIDATE_FLOORS[effort],8*(wanted+len(excluded))))
        nprobe=min(self.index.nlist,max(EFFORTS[effort],math.ceil(wanted/16)))
        native_started=time.perf_counter()
        with self.lock:
            self.index.nprobe=nprobe
            self.index.parallel_mode=1 if effort=='thorough' else 0
            _,found=self.index.search(query[None,:],candidates)
        diagnostics['candidate_search_seconds']=time.perf_counter()-native_started
        rescore_started=time.perf_counter()
        ids=np.unique(found[0]);ids=ids[(ids>=0)&(ids<self.count)]
        source_indices=np.searchsorted(self.starts,ids,side='right')-1
        groups,boundaries=np.unique(source_indices,return_index=True)
        boundaries=np.append(boundaries,len(ids))
        matches=[];rescored=0;sequential_reads=0
        # Thousands of small shard matrix-vector products are much faster with
        # one BLAS thread. Scope and restore this setting; keep Faiss OpenMP.
        controller=self.threadpool_controller()
        with self.lock,controller.limit():
            group_iter=iter(zip(groups,boundaries[:-1],boundaries[1:]))
            readers=min(8,max(1,self.manifest['parameters'].get('threads',8)),len(groups))
            if readers:
                with ThreadPoolExecutor(max_workers=readers,thread_name_prefix='candidate-reader',
                        initializer=lambda:controller.limit(limits=1,user_api='blas')) as pool:
                    pending=deque()
                    def submit():
                        group=next(group_iter,None)
                        if group is None:return False
                        source_index,lo,hi=group;source_index=int(source_index)
                        pending.append(pool.submit(self._score_source,source_index,
                            ids[lo:hi]-self.sources[source_index]['start'],query,wanted,excluded))
                        return True
                    for _ in range(readers):submit()
                    # Bounded futures keep at most one source per reader in flight.
                    while pending:
                        rows,count,sequential=pending.popleft().result()
                        matches.extend(rows);rescored+=count;sequential_reads+=int(sequential)
                        submit()
        diagnostics.update(backend='ivf-pq',approximate=True,nprobe=nprobe,effort=effort,
            rescore_seconds=time.perf_counter()-rescore_started,
            parallel_mode=1 if effort=='thorough' else 0,
            candidate_count=len(ids),rescored_count=rescored,reader_threads=readers,
            sequential_source_reads=sequential_reads,mapped_source_reads=len(groups)-sequential_reads,
            excluded_count=len(excluded),index_vectors=self.count,
            index_signature=self.manifest['snapshot_signature'])
        return sorted(matches,key=lambda m:(-m[2],m[0],m[1]))[:wanted]


@lru_cache(maxsize=1)
def cached_reader(directory,manifest_hash):
    path=Path(directory)/'search-index/manifest.json'
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=manifest_hash:
        raise ValueError('Search index changed during loading; retry the query')
    return IndexReader(Path(directory),json.loads(blob))


def clear_index_cache():
    cached_reader.cache_clear()


def indexed_search(directory,collection,query,top_k,effort='balanced',exclude_ids=None,diagnostics=None,progress=None):
    validate_options('ann',effort)
    if type(top_k) is not int or not 1<=top_k<=5000:
        raise ValueError('Indexed result count must be between 1 and 5,000')
    excluded=validate_exclusions(collection,exclude_ids)
    directory=Path(directory).resolve()
    path,manifest=read_manifest(directory,collection)
    diagnostics={} if diagnostics is None else diagnostics
    started=time.perf_counter()
    if progress:progress({'state':'loading_index','scanned_sources':0})
    reader=cached_reader(str(directory),sha256(path))
    # Reject changed original bundles even when the native index is cached.
    reader.verify_sources()
    diagnostics['index_load_seconds']=time.perf_counter()-started
    if progress:progress({'state':'searching','scanned_sources':0})
    result=reader.search(query,top_k,effort,excluded,diagnostics)
    diagnostics['search_seconds']=time.perf_counter()-started
    if progress:progress({'state':'metadata','scanned_sources':len(collection['shards'])})
    return result
