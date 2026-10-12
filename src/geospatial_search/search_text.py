"""Compatible indexed or exact text search; render only the top chip windows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import io
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from .bundles import selected_rows, top_matches, validate_bundle
from .download_nys_naip import Budget, sha256
from .embedding_control import gpu_access, idle_collection
from .model import encode_text, load_model
from .paths import EMBEDDINGS, IMAGERY, RESULTS, contained
from .tiling import read_rgb
from .search_index import IndexDependencyUnavailable,indexed_search,read_manifest,validate_exclusions,validate_options


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query',required=True)
    parser.add_argument('--collection-dir',type=Path,default=EMBEDDINGS)
    parser.add_argument('--imagery-dir',type=Path,default=IMAGERY)
    parser.add_argument('--output-dir',type=Path,default=RESULTS)
    parser.add_argument('--top-k',type=int,default=10)
    parser.add_argument('--model-cache-dir',type=Path)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--allow-partial',action='store_true',help='Search compatible published bundles from an unfinished run')
    parser.add_argument('--backend',choices=('auto','exact','ann'),default='auto')
    parser.add_argument('--search-effort',choices=('fast','balanced','thorough'),default='balanced')
    return parser.parse_args(argv)


def read_collection(directory: Path, allow_partial=False):
    root = json.loads((directory/'run.json').read_text(encoding='utf-8'))
    collection = json.loads((directory/'collection.json').read_text(encoding='utf-8'))
    if collection.get('run_fingerprint') != root.get('run_fingerprint'):
        raise ValueError('Collection and model/grid recipe do not match')
    if collection.get('state') not in ('complete','paused') and not allow_partial:
        raise ValueError('Embedding run is incomplete; use --allow-partial explicitly to search published bundles')
    if len({item['shard_id'] for item in collection['shards']}) != len(collection['shards']):
        raise ValueError('Duplicate embedding shard ID')
    if collection['chip_count'] != sum(item['chip_count'] for item in collection['shards']):
        raise ValueError('Collection chip counts differ')
    return root,collection


def exact_search_collection(directory: Path, collection: dict, query: np.ndarray, top_k: int):
    if top_k <= 0 or collection['chip_count'] == 0:
        raise ValueError('Require a positive top-k and at least one published vector')
    matches = []
    for entry in tqdm(collection['shards'],desc='Searching bundles',unit='tile'):
        shard = contained(directory/'shards',directory/'shards'/entry['shard_id'])
        run = validate_bundle(shard,collection['run_fingerprint'])
        if run['chip_count'] != entry['chip_count'] or run['source_uri'] != entry['source_uri']:
            raise ValueError('Bundle identity differs from collection catalog')
        vectors = np.load(shard/'embeddings.f16.npy',mmap_mode='r',allow_pickle=False)
        candidates = top_matches(vectors,query,top_k)
        matches.extend((entry['shard_id'],row_id,score) for row_id,score in candidates)
        matches = sorted(matches,key=lambda item:(-item[2],item[0],item[1]))[:top_k]
        del vectors
    return matches


def search_collection(directory: Path, collection: dict, query: np.ndarray, top_k: int,
                      *,backend='auto',effort='balanced',exclude_ids=None,diagnostics=None,progress=None):
    """Use a compatible ANN index when ready; retain the exact baseline/fallback."""
    import time
    validate_options(backend,effort)
    excluded=validate_exclusions(collection,exclude_ids)
    diagnostics={} if diagnostics is None else diagnostics
    started=time.perf_counter()
    if backend!='exact':
        try:
            read_manifest(Path(directory),collection)
        except (FileNotFoundError,ValueError) as exc:
            if backend=='ann':raise
            diagnostics['fallback_reason']=str(exc)
        else:
            try:
                return indexed_search(directory,collection,query,top_k,effort,excluded,diagnostics,progress)
            except IndexDependencyUnavailable as exc:
                if backend=='ann':raise
                diagnostics['fallback_reason']=str(exc)
    matches=exact_search_collection(directory,collection,query,min(collection['chip_count'],top_k+len(excluded)))
    matches=[m for m in matches if (m[0],m[1]) not in excluded][:top_k]
    diagnostics.update(backend='exact',approximate=False,search_seconds=time.perf_counter()-started,
                       effort=effort,excluded_count=len(excluded))
    return matches


def save_png(image, destination, budget):
    buffer = io.BytesIO()
    image.save(buffer,format='PNG')
    budget.write_bytes(destination,buffer.getvalue())


def render_chip(row, imagery_dir, rank, score, destination, budget):
    source = contained(imagery_dir,imagery_dir/row['source_uri'])
    with rasterio.Env(GDAL_PAM_ENABLED=False):
        with rasterio.open(source) as dataset:
            image = read_rgb(dataset,Window(row['pixel_x'],row['pixel_y'],row['pixel_width'],row['pixel_height']))
    labelled = Image.new('RGB',(image.width,image.height+38),'black')
    labelled.paste(image,(0,38))
    ImageDraw.Draw(labelled).text((8,10),f'#{rank} score={score:.4f}',fill='white')
    save_png(labelled,destination,budget)


def write_contact_sheet(paths, destination, budget):
    thumbnails = []
    for path in paths:
        with Image.open(path) as image:
            thumbnail = image.convert('RGB')
            thumbnail.thumbnail((320,320))
            thumbnails.append(thumbnail)
    columns = min(5,len(thumbnails))
    sheet = Image.new('RGB',(columns*320,((len(thumbnails)+columns-1)//columns)*320),'black')
    for index,image in enumerate(thumbnails):
        sheet.paste(image,((index%columns)*320,(index//columns)*320))
    save_png(sheet,destination,budget)


def run_search(args):
    if not 1 <= args.top_k <= 100:
        raise ValueError('Select between 1 and 100 rendered results')
    imagery_dir = args.imagery_dir.resolve()
    job_root = imagery_dir.parent
    results_root = contained(job_root,args.output_dir)
    for name in ('imagery','metadata','records','temporary'):
        protected = job_root/name
        if results_root == protected or results_root.is_relative_to(protected):
            raise ValueError('Search output overlaps protected source folders')
    budget = Budget(job_root)
    directory = args.collection_dir.resolve()
    root,collection = read_collection(directory,args.allow_partial)
    if collection['chip_count']==0:
        raise ValueError('No saved embeddings yet. Resume embedding until at least one source tile has been saved, then pause again.')
    print(f"Searching {collection['chip_count']:,} saved embeddings from {collection.get('completed_sources',len(collection['shards']))}/{collection.get('total_sources','?')} source tiles ({collection['state']}).",flush=True)
    stored = root['model_contract']
    config = {**root['recipe']['model'],'amp':stored['amp']}
    model,_,device,loaded = load_model(config,cache_dir=args.model_cache_dir,offline=args.offline)
    for name in ('model_id','model_revision','arch','model_fingerprint','preprocess_fingerprint','embedding_dim'):
        if loaded[name] != stored[name]:
            raise ValueError(f'Query model differs from image embeddings: {name}')
    query = encode_text(model,args.query,config['arch'],device,config['amp'])
    search_details={}
    matches = search_collection(directory,collection,query,args.top_k,backend=args.backend,
                                effort=args.search_effort,diagnostics=search_details)
    print(f"Retrieval: {search_details['backend']} in {search_details['search_seconds']:.2f}s",flush=True)
    by_shard = {}
    for shard_id,row_id,_ in matches:
        by_shard.setdefault(shard_id,[]).append(row_id)
    metadata = {shard:selected_rows(directory/'shards'/shard/'chips.geoparquet',ids) for shard,ids in by_shard.items()}
    safe_query = re.sub(r'[^a-z0-9]+','-',args.query.lower()).strip('-')[:48] or 'query'
    destination = results_root/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{safe_query}'
    destination.mkdir(parents=True,exist_ok=False)
    pngs,results = [],[]
    fields = ('embedding_id','chip_id','source_uri','source_sha256','pixel_x','pixel_y','pixel_width','pixel_height',
              'centroid_lon','centroid_lat','west','south','east','north','crs_wkt','scale_id')
    verified_sources = set()
    for rank,(shard,row_id,score) in enumerate(matches,1):
        row = metadata[shard][row_id]
        source = contained(imagery_dir,imagery_dir/row['source_uri'])
        if source not in verified_sources:
            if sha256(source) != row['source_sha256']:
                raise ValueError(f'Result source checksum differs: {source}')
            verified_sources.add(source)
        path = destination/f'rank-{rank:02d}_score-{score:.4f}.png'
        render_chip(row,imagery_dir,rank,score,path,budget)
        pngs.append(path)
        results.append({'rank':rank,'score':score,'shard_id':shard,**{field:row[field] for field in fields},'png':path.name})
        print(f'#{rank:02d} {score:.4f} {row["chip_id"]}')
    write_contact_sheet(pngs,destination/'contact-sheet.png',budget)
    budget.atomic_json(destination/'results.json',{'query':args.query,'run_fingerprint':collection['run_fingerprint'],
        'scope':collection['scope'],'collection_state':collection['state'],
        'partial_collection':collection['state']!='complete','searched_chip_count':collection['chip_count'],
        'searched_source_tiles':len(collection['shards']),'total_source_tiles':collection.get('total_sources'),
        'collection_dir':str(directory),'search':search_details,'results':results})
    print(f'Results: {destination}')
    return 0


def main(argv=None):
    args=parse_args(argv)
    directory=args.collection_dir.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f'Embedding collection does not exist: {directory}')
    with idle_collection(directory), gpu_access(args.imagery_dir.resolve().parent):
        try:
            return run_search(args)
        finally:
            # run_search's local model has been released before unlocking.
            import sys
            import gc
            gc.collect()
            torch=sys.modules.get('torch')
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
