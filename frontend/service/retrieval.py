"""Retrieval boundary: read snapshots and use compatible indexed or exact search."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from contextlib import ExitStack, nullcontext

from storage import atomic_json, contained

MODEL_KEYS=('model_id','model_revision','arch','model_fingerprint','preprocess_fingerprint','embedding_dim')
BUSY_MESSAGE='Embedding is running; pause it using the existing command before searching. Run .\\scripts\\embed-nys.ps1 -Pause yourself and wait for acknowledgement.'


def lock_held(path):
    """Read-only lock probe. Never creates or updates a backend lock file."""
    try:handle=Path(path).open('rb')
    except FileNotFoundError:return False
    try:
        if os.name=='nt':
            import msvcrt
            try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:return True
            msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:return True
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        return False
    finally:handle.close()


def validate_query(value):
    query=value.get('query');top_k=value.get('top_k',25);allow_partial=value.get('allow_partial',False)
    if not isinstance(query,str) or not 1<=len(query.strip())<=200:raise ValueError('Enter a phrase of 1 to 200 characters')
    if type(top_k) is not int or not 1<=top_k<=5000:raise ValueError('Result count must be between 1 and 5,000')
    if type(allow_partial) is not bool:raise ValueError('allow_partial must be boolean')
    return query.strip(),top_k,allow_partial


def verify_model_contract(stored,loaded):
    for name in MODEL_KEYS:
        if loaded.get(name)!=stored.get(name):raise ValueError(f'Query model differs from image embeddings: {name}')


def validate_search_options(value):
    from geospatial_search.search_index import validate_options
    backend=value.get('search_backend','auto');effort=value.get('search_effort','balanced')
    validate_options(backend,effort)
    seen=value.get('exclude_ids',[])
    if not isinstance(seen,list) or len(seen)>20000:raise ValueError('Invalid already-seen chip IDs')
    for pair in seen:
        if (not isinstance(pair,list) or len(pair) not in (2,3)
            or any(not isinstance(value,str) or not 1<=len(value)<=160 for value in pair[:-1])
            or type(pair[-1]) is not int or pair[-1]<0):
            raise ValueError('Excluded IDs must be [shard_id, embedding_id] or [dataset_id, shard_id, embedding_id]')
    return backend,effort,seen


def status_snapshot(collection,job,include_supplements=True):
    collection=Path(collection);job=Path(job)
    try:
        root=json.loads((collection/'run.json').read_text(encoding='utf-8'))
        catalog=json.loads((collection/'collection.json').read_text(encoding='utf-8'))
        if root.get('run_fingerprint')!=catalog.get('run_fingerprint'):raise ValueError('Collection fingerprint differs')
        shards=catalog.get('shards',[])
        if len({s['shard_id'] for s in shards})!=len(shards) or catalog['chip_count']!=sum(s['chip_count'] for s in shards):raise ValueError('Published catalog is inconsistent')
    except FileNotFoundError:
        catalog={'shards':[],'chip_count':0,'state':'missing'};shards=[]
    try:worker=json.loads((collection/'status.json').read_text(encoding='utf-8'))
    except FileNotFoundError:worker={}
    collection_busy=lock_held(collection/'worker.lock')
    gpu_busy=lock_held(job/'embeddings/gpu-runtime/worker.lock')
    from geospatial_search.search_index import describe_index
    index=describe_index(collection,catalog)
    coverage=[s['source_uri'] for s in shards];total_sources=catalog.get('total_sources',3805)
    published=len(shards);chips=catalog['chip_count'];states=[catalog.get('state')]
    from geospatial_search.gapfill import collections
    for dataset in collections(job,include_supplements)[1:]:
        extra=json.loads((dataset.directory/'collection.json').read_text(encoding='utf-8'))
        extra_root=json.loads((dataset.directory/'run.json').read_text(encoding='utf-8'))
        if extra['run_fingerprint']!=extra_root['run_fingerprint']:raise ValueError('Supplement collection fingerprint differs')
        coverage.extend(dataset.qualify(s['source_uri']) for s in extra['shards'])
        published+=len(extra['shards']);chips+=extra['chip_count'];total_sources+=extra['total_sources'];states.append(extra['state'])
        if extra['state']!='complete' and lock_held(dataset.directory/'worker.lock'):
            worker=json.loads((dataset.directory/'status.json').read_text(encoding='utf-8'))
    collection_state='complete' if all(state=='complete' for state in states) else ('paused' if all(state in ('complete','paused') for state in states) else catalog.get('state'))
    return {'state':worker.get('state',catalog.get('state','missing')),'collection_state':collection_state,
            'busy':collection_busy or gpu_busy,'collection_busy':collection_busy,'gpu_busy':gpu_busy,
            'published_sources':published,'total_sources':total_sources,
            'published_chips':chips,'partial':any(state!='complete' for state in states),
            'run_fingerprint':catalog.get('run_fingerprint'),'elapsed_hms':worker.get('elapsed_hms'),
            'eta_hms':worker.get('eta_hms'),'percent_complete':worker.get('percent_complete'),
            'coverage':coverage,'search_index':index}


class ExactRetrieval:
    """Compatible retrieval boundary; indexed search with the exact fallback.

    Similar-chip and geographic-neighbor operations can be added here without
    changing the UI. The legacy class name remains for worker compatibility.
    """
    def __init__(self,repository,frontend):
        self.repository=Path(repository).resolve();self.frontend=Path(frontend).resolve()
        self.job=self.repository/'data/nys-naip-2022-rgb-060m'
        self.collection=self.job/'embeddings/georsclip-vith14-s256-st192'
        self.resident_model=None;self.resident_device=None;self.resident_contract=None
        self.resident_fingerprint=None;self.resident_locks=None;self.resident_collections=set()

    def preload(self,allow_partial=False,offline=True,progress=lambda value:None):
        """Keep the pinned model and both OS locks until explicitly closed."""
        if self.resident_locks is not None:return
        from geospatial_search.embedding_control import idle_collection,gpu_access
        from geospatial_search import search_text
        from geospatial_search.model import load_model,encode_text
        if status_snapshot(self.collection,self.job)['busy']:raise RuntimeError(BUSY_MESSAGE)
        self.resident_locks=ExitStack()
        try:
            self.resident_locks.enter_context(idle_collection(self.collection))
            self.resident_locks.enter_context(gpu_access(self.job))
            from geospatial_search.gapfill import collections
            from geospatial_search.federated import compatible_models
            root,collection=search_text.read_collection(self.collection,allow_partial)
            self.resident_collections={self.collection.resolve()}
            for dataset in collections(self.job)[1:]:
                self.resident_locks.enter_context(idle_collection(dataset.directory))
                supplement,_=search_text.read_collection(dataset.directory,allow_partial)
                compatible_models(root['model_contract'],supplement['model_contract'])
                self.resident_collections.add(dataset.directory.resolve())
            if not collection['chip_count']:raise ValueError('No saved embeddings yet')
            progress({'state':'loading'})
            stored=root['model_contract'];config={**root['recipe']['model'],'amp':stored['amp']}
            self.resident_model,_,self.resident_device,self.resident_contract=load_model(config,offline=offline)
            verify_model_contract(stored,self.resident_contract)
            self.resident_fingerprint=root['run_fingerprint']
            # Initialize text inference now, rather than charging the first query.
            encode_text(self.resident_model,'aerial imagery',config['arch'],self.resident_device,config['amp'])
            progress({'state':'ready','run_fingerprint':self.resident_fingerprint,'gpu':self.resident_contract.get('gpu')})
        except Exception:
            self.close();raise

    def close(self):
        self.resident_model=None;self.resident_device=None;self.resident_contract=None;self.resident_fingerprint=None
        self.resident_collections=set()
        try:
            gc.collect()
            torch=sys.modules.get('torch')
            if torch is not None and torch.cuda.is_available():torch.cuda.empty_cache()
        finally:
            if self.resident_locks is not None:
                self.resident_locks.close();self.resident_locks=None

    def text(self,request,progress):
        query,top_k,allow_partial=validate_query(request)
        backend,effort,seen=validate_search_options(request)
        from geospatial_search import search_text
        from geospatial_search.embedding_control import idle_collection,gpu_access
        from geospatial_search.model import load_model,encode_text
        from geospatial_search.bundles import selected_rows
        from geospatial_search.download_nys_naip import sha256
        resident=self.resident_locks is not None
        if not resident and status_snapshot(self.collection,self.job)['busy']:raise RuntimeError(BUSY_MESSAGE)
        with (nullcontext() if resident else idle_collection(self.collection)), (nullcontext() if resident else gpu_access(self.job)), ExitStack() as extra_locks:
            model=None;query_vector=None
            try:
                root,collection=search_text.read_collection(self.collection,allow_partial)
                if resident:
                    if root['run_fingerprint']!=self.resident_fingerprint:raise ValueError('Loaded GPU model belongs to another collection fingerprint; unload it first')
                    verify_model_contract(root['model_contract'],self.resident_contract)
                if not collection['chip_count']:raise ValueError('No saved embeddings yet')
                from geospatial_search.gapfill import Collection,BASE_DATASET,collections
                from geospatial_search.federated import compatible_models,search as federated_search
                if type(request.get('include_supplements',True)) is not bool:raise ValueError('include_supplements must be boolean')
                datasets=[Collection(root['recipe'].get('dataset_id',BASE_DATASET),self.collection,self.job/'imagery',2022)]
                datasets += collections(self.job,request.get('include_supplements',True))[1:]
                entries=[(datasets[0],root,collection)]
                for dataset in datasets[1:]:
                    if resident:
                        if dataset.directory.resolve() not in self.resident_collections:
                            self.resident_locks.enter_context(idle_collection(dataset.directory));self.resident_collections.add(dataset.directory.resolve())
                    else:extra_locks.enter_context(idle_collection(dataset.directory))
                    other_root,catalog=search_text.read_collection(dataset.directory,allow_partial)
                    compatible_models(root['model_contract'],other_root['model_contract'])
                    entries.append((dataset,other_root,catalog))
                total=sum(len(catalog['shards']) for _,_,catalog in entries)
                total_sources=sum(catalog.get('total_sources',3805) for _,_,catalog in entries)
                progress({'state':'encoding','scanned_sources':0,'searched_source_tiles':total,'total_source_tiles':collection.get('total_sources',3805)})
                stored=root['model_contract'];config={**root['recipe']['model'],'amp':stored['amp']}
                key=hashlib.sha256((root['run_fingerprint']+'\0'+query).encode()).hexdigest()
                import numpy as np
                vector_path=contained(self.frontend,self.frontend/'state/text-vectors'/f'{key}.json')
                if vector_path.exists():
                    saved=json.loads(vector_path.read_text(encoding='utf-8'))
                    if saved['run_fingerprint']!=root['run_fingerprint']:raise ValueError('Cached query fingerprint differs')
                    query_vector=np.asarray(saved['vector'],dtype=np.float32)
                    if query_vector.shape!=(stored['embedding_dim'],) or not np.isfinite(query_vector).all() or not np.isclose(np.linalg.norm(query_vector),1,atol=.002):raise ValueError('Invalid cached query vector')
                else:
                    if resident:model=self.resident_model;device=self.resident_device
                    else:
                        model,_,device,loaded=load_model(config,offline=request.get('offline_model',True))
                        verify_model_contract(stored,loaded)
                    query_vector=encode_text(model,query,config['arch'],device,config['amp'])
                    atomic_json(vector_path,{'run_fingerprint':root['run_fingerprint'],'vector':query_vector.tolist()},self.frontend)
                matches,search_details=federated_search(entries,query_vector,top_k,
                    backend=backend,effort=effort,exclude_ids=seen,
                    progress=lambda value:progress({'searched_source_tiles':total,'total_source_tiles':collection.get('total_sources',3805),**value}))
                by_shard={}
                for dataset,shard,row_id,_ in matches:by_shard.setdefault((dataset.dataset_id,shard),[]).append(row_id)
                by_dataset={dataset.dataset_id:dataset for dataset in datasets}
                metadata={(dataset_id,shard):selected_rows(contained(by_dataset[dataset_id].directory/'shards',by_dataset[dataset_id].directory/'shards'/shard)/'chips.geoparquet',ids)
                    for (dataset_id,shard),ids in by_shard.items()}
                from shapely import from_wkb
                verified=set();results=[]
                for rank,(dataset,shard,row_id,score) in enumerate(matches,1):
                    row=metadata[(dataset.dataset_id,shard)][row_id]
                    source=contained(dataset.imagery,dataset.imagery/row['source_uri'])
                    if source not in verified:
                        if sha256(source)!=row['source_sha256']:raise ValueError('Result source checksum differs')
                        verified.add(source)
                    fields=('embedding_id','chip_id','source_uri','source_sha256','pixel_x','pixel_y','pixel_width','pixel_height','centroid_lon','centroid_lat','west','south','east','north','crs_wkt','scale_id','window_transform')
                    results.append({'rank':rank,'score':score,'shard_id':shard,**{name:row[name] for name in fields},
                        'dataset_id':dataset.dataset_id,'dataset_source_uri':row['source_uri'],'source_uri':dataset.qualify(row['source_uri']),
                        'release_year':dataset.release_year,'acquisition_date':dataset.acquisition_date(row['source_uri']),
                        'geometry':from_wkb(row['geometry']).__geo_interface__})
                return {'query':query,'run_fingerprint':collection['run_fingerprint'],'collection_state':collection['state'],
                        'partial_collection':any(catalog['state']!='complete' for _,_,catalog in entries),'searched_source_tiles':total,'total_source_tiles':total_sources,
                        'collection_fingerprints':{dataset.dataset_id:recipe['run_fingerprint'] for dataset,recipe,_ in entries},
                        'searched_chip_count':sum(catalog['chip_count'] for _,_,catalog in entries),
                        'searched_source_uris':[dataset.qualify(s['source_uri']) for dataset,_,catalog in entries for s in catalog['shards']],
                        'created_utc':time.time(),'search':search_details,'results':results}
            finally:
                model=None;query_vector=None
                if not resident:
                    gc.collect()
                    torch=sys.modules.get('torch')
                    if torch is not None and torch.cuda.is_available():torch.cuda.empty_cache()
