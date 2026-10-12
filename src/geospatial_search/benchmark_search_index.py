"""Measure current exact retrieval against an index on identical real queries."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from .download_nys_naip import Budget
from .paths import JOB,EMBEDDINGS,contained
from .search_text import read_collection,search_collection
from .search_index import snapshot_signature,EFFORTS,CANDIDATE_FLOORS
from .bundles import validate_bundle,top_matches
from tqdm import tqdm


def shared_exact_references(directory,catalog,queries,top_k=500):
    """Reuse one source validation/read across queries; keep original scoring."""
    matches=[[] for _ in queries]
    for entry in tqdm(catalog['shards'],desc='Exact reference queries',unit='tile'):
        shard=contained(directory/'shards',directory/'shards'/entry['shard_id'])
        run=validate_bundle(shard,catalog['run_fingerprint'])
        if run['chip_count']!=entry['chip_count'] or run['source_uri']!=entry['source_uri']:
            raise ValueError('Bundle identity differs from collection catalog')
        vectors=np.load(shard/'embeddings.f16.npy',mmap_mode='r',allow_pickle=False)
        for index,query in enumerate(queries):
            candidates=top_matches(vectors,query,top_k)
            matches[index].extend((entry['shard_id'],row,score) for row,score in candidates)
            matches[index]=sorted(matches[index],key=lambda m:(-m[2],m[0],m[1]))[:top_k]
        del vectors
    return matches


def benchmark(directory,queries,output,job_root=JOB,timed_exact_queries=3,recompute_references=False):
    root,catalog=read_collection(directory)
    if queries['run_fingerprint']!=root['run_fingerprint']:
        raise ValueError('Benchmark queries belong to another model/run')
    report={'run_fingerprint':root['run_fingerprint'],'vectors':catalog['chip_count'],
        'dimension':catalog['embedding_dim'],'source_tiles':len(catalog['shards']),
        'snapshot_signature':snapshot_signature(catalog),
        'queries_sha256':hashlib.sha256(json.dumps(queries,sort_keys=True).encode()).hexdigest(),
        'measurement':'retrieval only; text encoding and PNG rendering excluded',
        'nprobe_defaults':EFFORTS,'candidate_floors':CANDIDATE_FLOORS,'queries':[]}
    previous=json.loads(output.read_text()) if output.exists() and not recompute_references else {}
    reference={q['query']:q for q in previous.get('queries',[]) if q.get('reference_matches')}
    if (previous.get('snapshot_signature')!=report['snapshot_signature']
        or previous.get('queries_sha256')!=report['queries_sha256']):
        reference={}
    budget=Budget(job_root)
    # Time independent original scans first. Remaining recall references use
    # the identical ranking helper with shared bundle validation and disk reads.
    for query in queries['queries'][:timed_exact_queries]:
        if query['query'] in reference and reference[query['query']]['exact_seconds'] is not None:continue
        start=time.perf_counter()
        truth=search_collection(directory,catalog,np.asarray(query['vector'],dtype=np.float32),500,backend='exact')
        entry={'query':query['query'],'exact_seconds':time.perf_counter()-start,
               'reference_matches':truth,'trials':[]}
        reference[query['query']]=entry
        report['queries']=[reference[q['query']] for q in queries['queries'] if q['query'] in reference]
        budget.atomic_json(output,report)
    remaining=[q for q in queries['queries'] if q['query'] not in reference]
    if remaining:
        start=time.perf_counter()
        truths=shared_exact_references(directory,catalog,[np.asarray(q['vector'],dtype=np.float32) for q in remaining])
        report['shared_reference_seconds']=time.perf_counter()-start
        for query,truth in zip(remaining,truths):
            entry={'query':query['query'],'exact_seconds':None,'reference_matches':truth,'trials':[]}
            reference[query['query']]=entry;report['queries'].append(entry)
        budget.atomic_json(output,report)
    report['queries']=[reference[q['query']] for q in queries['queries']]
    for entry in report['queries']:entry['trials']=[]
    budget.atomic_json(output,report)
    for number,query in enumerate(queries['queries'],1):
        vector=np.asarray(query['vector'],dtype=np.float32)
        saved=reference[query['query']]
        truth=saved['reference_matches'];exact_seconds=saved['exact_seconds']
        entry=report['queries'][number-1]
        for effort in ('fast','balanced','thorough'):
            details={};start=time.perf_counter()
            found=search_collection(directory,catalog,vector,500,backend='ann',effort=effort,diagnostics=details)
            seconds=time.perf_counter()-start
            entry['trials'].append({'effort':effort,'seconds':seconds,**details,
                'recall':{str(k):len({(s,i) for s,i,_ in found[:k]}&{(s,i) for s,i,_ in truth[:k]})/min(k,len(truth))
                          for k in (20,100,500)}})
        budget.atomic_json(output,report)
        exact_label=f'{exact_seconds:.2f}s' if exact_seconds is not None else 'shared reference'
        print(f"{number}/{len(queries['queries'])} {query['query']}: exact {exact_label}; "+
            '; '.join(f"{t['effort']} {t['seconds']:.2f}s recall@100 {t['recall']['100']:.3f}" for t in entry['trials']),flush=True)
    timed=[q['exact_seconds'] for q in report['queries'] if q['exact_seconds'] is not None]
    report['summary']={'exact_median_seconds':statistics.median(timed) if timed else None,
                       'independent_exact_timings':len(timed),'recall_query_count':len(report['queries'])}
    for effort in ('fast','balanced','thorough'):
        trials=[next(t for t in q['trials'] if t['effort']==effort) for q in report['queries']]
        report['summary'][effort]={'median_seconds':statistics.median(t['seconds'] for t in trials),
            'p95_seconds':float(np.percentile([t['seconds'] for t in trials],95)),
            'recall':{str(k):{'mean':float(np.mean([t['recall'][str(k)] for t in trials])),
                              'minimum':min(t['recall'][str(k)] for t in trials)} for k in (20,100,500)}}
    budget.atomic_json(output,report)
    print(json.dumps(report['summary'],indent=2),flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-dir',type=Path,default=EMBEDDINGS)
    parser.add_argument('--queries',type=Path,default=JOB/'verification/search-selection/text-queries.json')
    parser.add_argument('--output',type=Path,default=JOB/'verification/search-selection/full-benchmark.json')
    parser.add_argument('--timed-exact-queries',type=int,default=3)
    parser.add_argument('--recompute-references',action='store_true',help='Remeasure exact scans instead of reusing saved local references')
    args=parser.parse_args(argv)
    if args.timed_exact_queries<0:parser.error('Timed exact query count must be nonnegative')
    benchmark(args.collection_dir,json.loads(args.queries.read_text()),contained(JOB,args.output),
        timed_exact_queries=args.timed_exact_queries,recompute_references=args.recompute_references)
    return 0


if __name__=='__main__':raise SystemExit(main())
