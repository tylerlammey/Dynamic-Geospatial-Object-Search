"""Build a checksummed, storage-bounded CPU IVF-PQ index from published bundles."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from .bundles import validate_bundle
from .download_nys_naip import Budget,job_lock,sha256
from .paths import EMBEDDINGS,JOB,contained
from .search_index import file_stat,snapshot_signature


def build_index(directory,job_root,*,partitions=4096,pq_subvectors=128,training_limit=327680,
                iterations=15,threads=8,seed=20261012,force=False):
    import faiss
    from .search_text import read_collection
    directory=contained(job_root,Path(directory));output=directory/'search-index'
    for name in ('imagery','records','metadata','temporary'):
        protected=(Path(job_root)/name).resolve()
        if directory==protected or directory.is_relative_to(protected):
            raise ValueError('Search index cannot overlap protected imagery/provenance folders')
    output.mkdir(exist_ok=True)
    with job_lock(output):
        root,collection=read_collection(directory)
        if collection['state'] not in ('complete','paused') or not collection['chip_count']:
            raise ValueError('Build from a complete or safely paused published collection')
        signature=snapshot_signature(collection)
        existing=output/'manifest.json'
        if existing.exists() and not force:
            saved=json.loads(existing.read_text())
            if saved['snapshot_signature']==signature:
                print('Compatible search index already exists.',flush=True)
                return saved
            raise FileExistsError('Index covers another snapshot; use --force to publish a new generation')
        dimension=collection['embedding_dim'];total=collection['chip_count']
        if (not 1<=partitions<=65536 or not 1<=pq_subvectors<=dimension or dimension%pq_subvectors
            or not 1<=threads<=16 or not 1<=iterations<=100 or training_limit<256 or total<256):
            raise ValueError('Invalid IVF/PQ training parameters or too few vectors')
        partitions=min(partitions,max(1,min(total,training_limit)//64))
        budget=Budget(job_root);sources=[];offset=0;started=time.perf_counter();building_started=None
        def status(state,processed=0,error=None):
            elapsed=time.perf_counter()-started
            rate=processed/max(.001,time.perf_counter()-building_started) if processed and building_started else None
            budget.atomic_json(output/'status.json',{'state':state,'processed_vectors':processed,'total_vectors':total,
                'percent':100*processed/total,'elapsed_seconds':elapsed,'eta_seconds':(total-processed)/rate if rate else None,
                'error':error,'pid':__import__('os').getpid(),'updated_utc':datetime.now(timezone.utc).isoformat()})
        try:
            status('validating')
            for entry in tqdm(collection['shards'],desc='Validate index sources',unit='tile'):
                shard=contained(directory/'shards',directory/'shards'/entry['shard_id'])
                run=validate_bundle(shard,collection['run_fingerprint'],verify_hashes=True)
                if run['chip_count']!=entry['chip_count'] or run['source_uri']!=entry['source_uri']:
                    raise ValueError('Published source identity differs')
                sources.append({'shard_id':entry['shard_id'],'source_uri':entry['source_uri'],
                    'start':offset,'count':entry['chip_count'],
                    'files':{name:file_stat(shard/name) for name in ('embeddings.f16.npy','chips.geoparquet','run.json')}})
                offset+=entry['chip_count']
            if offset!=total:raise ValueError('Collection vector count differs')
            # Uniform global-row sample; larger source bundles contribute more rows.
            limit=min(training_limit,total);rng=np.random.default_rng(seed)
            sample_ids=np.sort(rng.choice(total,limit,replace=False))
            sample=np.empty((limit,dimension),dtype=np.float32)
            for source in sources:
                lo=np.searchsorted(sample_ids,source['start']);hi=np.searchsorted(sample_ids,source['start']+source['count'])
                if hi>lo:
                    values=np.load(directory/'shards'/source['shard_id']/'embeddings.f16.npy',mmap_mode='r')
                    sample[lo:hi]=values[sample_ids[lo:hi]-source['start']]
                    del values
            if not np.isfinite(sample).all() or not np.allclose(np.linalg.norm(sample,axis=1),1,atol=.002):
                raise ValueError('Invalid normalized training vectors')
            faiss.omp_set_num_threads(threads)
            coarse=faiss.IndexFlatIP(dimension)
            index=faiss.IndexIVFPQ(coarse,dimension,partitions,pq_subvectors,8,faiss.METRIC_INNER_PRODUCT)
            index.cp.niter=iterations;index.cp.seed=seed
            index.pq.cp.niter=iterations;index.pq.cp.seed=seed
            index.cp.min_points_per_centroid=1;index.pq.cp.min_points_per_centroid=1
            index.use_precomputed_table=-1
            status('training');print(f'Training IVF-PQ on {limit:,} vectors; {partitions:,} partitions, {pq_subvectors}-byte codes.',flush=True)
            index.train(sample);del sample
            building_started=time.perf_counter()
            progress=tqdm(total=total,desc='Build search index',unit='vector',mininterval=2)
            last_status=0
            for source in sources:
                values=np.load(directory/'shards'/source['shard_id']/'embeddings.f16.npy',mmap_mode='r')
                for start in range(0,len(values),16384):
                    block=values[start:start+16384].astype(np.float32)
                    if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block,axis=1),1,atol=.002):
                        raise ValueError('Invalid normalized source vectors')
                    index.add(block);progress.update(len(block))
                del values
                if time.perf_counter()-last_status>2:
                    status('building',index.ntotal);last_status=time.perf_counter()
            progress.close()
            if index.ntotal!=total:raise ValueError('Native vector count differs')
            # Verify the snapshot and its immutable files again before publication.
            current=json.loads((directory/'collection.json').read_text())
            if snapshot_signature(current)!=signature:raise ValueError('Collection changed during index build')
            for source in sources:
                for name,stat in source['files'].items():
                    if file_stat(directory/'shards'/source['shard_id']/name)!=stat:
                        raise ValueError('An embedding bundle changed during index build')
            status('publishing',total)
            serialized=faiss.serialize_index(index)
            stage=output/'index.faiss.new'
            budget.write_bytes(stage,serialized)
            checksum=sha256(stage);name=f'index-{checksum[:20]}.faiss'
            destination=output/name
            budget.replace(stage,destination)
            manifest={'format_version':1,'algorithm':'ivf-pq','snapshot_signature':signature,
                'run_fingerprint':collection['run_fingerprint'],'dimension':dimension,'ntotal':total,
                'metric':'original normalized-vector inner product','score_policy':'exact float32 dot product on original float16 candidates',
                'index_file':name,'index_sha256':checksum,'index_bytes':len(serialized),'faiss_version':faiss.__version__,
                'parameters':{'partitions':partitions,'pq_subvectors':pq_subvectors,'bits':8,'training_vectors':limit,
                    'iterations':iterations,'seed':seed,'threads':threads},'sources':sources,
                'created_utc':datetime.now(timezone.utc).isoformat(),'build_seconds':time.perf_counter()-started}
            budget.atomic_json(existing,manifest)
            status('complete',total)
            print(f'Index ready: {total:,} vectors; {len(serialized)/1e9:.2f} GB; {manifest["build_seconds"]:.1f}s. {output}',flush=True)
            return manifest
        except BaseException as exc:
            status('failed',error=str(exc));raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-dir',type=Path,default=EMBEDDINGS)
    parser.add_argument('--job-root',type=Path,default=JOB)
    parser.add_argument('--partitions',type=int,default=4096)
    parser.add_argument('--pq-subvectors',type=int,default=128)
    parser.add_argument('--training-vectors',type=int,default=327680)
    parser.add_argument('--iterations',type=int,default=15)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args(argv)
    build_index(args.collection_dir,args.job_root,partitions=args.partitions,pq_subvectors=args.pq_subvectors,
                training_limit=args.training_vectors,iterations=args.iterations,threads=args.threads,force=args.force)
    return 0


if __name__=='__main__':raise SystemExit(main())
