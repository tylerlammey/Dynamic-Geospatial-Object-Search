"""Encode once, retrieve distinct compatible collections, merge cosine scores."""
import time
from .search_index import validate_exclusions

MODEL_KEYS=('model_id','model_revision','arch','model_fingerprint','preprocess_fingerprint',
            'embedding_dim','metric','embedding_dtype','amp','rgb_band_order')


def compatible_models(first,other):
    for name in MODEL_KEYS:
        if first.get(name)!=other.get(name):raise ValueError(f'Collection query-model contract differs: {name}')


def search(entries,query,top_k,*,backend='auto',effort='balanced',exclude_ids=None,progress=None):
    from .search_text import search_collection
    started=time.perf_counter();by_dataset={entry[0].dataset_id:[] for entry in entries}
    if not entries:raise ValueError('No searchable collections')
    for identifier in exclude_ids or []:
        if len(identifier)==2:dataset=entries[0][0].dataset_id;pair=identifier
        elif len(identifier)==3:dataset=identifier[0];pair=identifier[1:]
        else:raise ValueError('Seen IDs require [dataset_id, shard_id, embedding_id]')
        if dataset not in by_dataset:raise ValueError('Excluded dataset is outside the searched snapshot')
        by_dataset[dataset].append(pair)
    for dataset,root,catalog in entries:validate_exclusions(catalog,by_dataset[dataset.dataset_id])
    merged=[];diagnostics=[]
    for priority,(dataset,root,catalog) in enumerate(entries):
        details={}
        # A small supplement can scan its original vectors without building a
        # second approximate index. Explicit ANN still applies to the base.
        selected_backend=backend if priority==0 or backend!='ann' else 'auto'
        matches=search_collection(dataset.directory,catalog,query,top_k,backend=selected_backend,effort=effort,
            exclude_ids=by_dataset[dataset.dataset_id],diagnostics=details,progress=progress)
        merged.extend((dataset,shard,row,float(score),priority) for shard,row,score in matches)
        diagnostics.append({'dataset_id':dataset.dataset_id,**details})
    merged.sort(key=lambda value:(-value[3],value[4],value[1],value[2]))
    approximate=any(details.get('approximate',False) for details in diagnostics)
    return [value[:4] for value in merged[:top_k]],{
        'backend':diagnostics[0].get('backend','exact') if len(entries)==1 else 'federated',
        'approximate':approximate,'search_seconds':time.perf_counter()-started,'effort':effort,'collections':diagnostics}
