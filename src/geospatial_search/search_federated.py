"""CLI search over the frozen 2022 collection and its labeled gap supplement."""
from contextlib import ExitStack
from datetime import datetime,timezone
import gc
import re
import sys

from . import search_text
from .bundles import selected_rows
from .download_nys_naip import Budget,sha256
from .embedding_control import gpu_access,idle_collection
from .federated import compatible_models,search
from .gapfill import collections
from .model import load_model,encode_text
from .paths import contained


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    base_only='--base-only' in arguments
    if base_only:arguments.remove('--base-only')
    args=search_text.parse_args(arguments);job=args.imagery_dir.resolve().parent
    if not 1<=args.top_k<=100:raise ValueError('Select between 1 and 100 rendered results')
    datasets=collections(job,not base_only)
    with ExitStack() as locks:
        for dataset in datasets:locks.enter_context(idle_collection(dataset.directory))
        locks.enter_context(gpu_access(job));model=None
        try:
            entries=[(dataset,*search_text.read_collection(dataset.directory,args.allow_partial)) for dataset in datasets]
            root=entries[0][1];contract=root['model_contract']
            for _,other,_ in entries[1:]:compatible_models(contract,other['model_contract'])
            config={**root['recipe']['model'],'amp':contract['amp']}
            model,_,device,loaded=load_model(config,cache_dir=args.model_cache_dir,offline=args.offline)
            compatible_models(contract,loaded)
            query=encode_text(model,args.query,config['arch'],device,config['amp'])
            matches,details=search(entries,query,args.top_k,backend=args.backend,effort=args.search_effort)
            budget=Budget(job);output=contained(job,args.output_dir.resolve())
            phrase=re.sub(r'[^a-z0-9]+','-',args.query.lower()).strip('-')[:48] or 'query'
            destination=output/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{phrase}'
            destination.mkdir(parents=True,exist_ok=False);by_shard={}
            for dataset,shard,row,score in matches:by_shard.setdefault((dataset.dataset_id,shard),[]).append(row)
            by_dataset={dataset.dataset_id:dataset for dataset in datasets}
            metadata={(dataset_id,shard):selected_rows(contained(by_dataset[dataset_id].directory/'shards',by_dataset[dataset_id].directory/'shards'/shard)/'chips.geoparquet',ids)
                for (dataset_id,shard),ids in by_shard.items()}
            results=[];pngs=[];verified=set()
            fields=('embedding_id','chip_id','source_sha256','pixel_x','pixel_y','pixel_width','pixel_height',
                'centroid_lon','centroid_lat','west','south','east','north','crs_wkt','scale_id')
            for rank,(dataset,shard,row_id,score) in enumerate(matches,1):
                row=metadata[(dataset.dataset_id,shard)][row_id];source=contained(dataset.imagery,dataset.imagery/row['source_uri'])
                if source not in verified:
                    if sha256(source)!=row['source_sha256']:raise ValueError('Matched source checksum differs')
                    verified.add(source)
                path=destination/f'rank-{rank:02d}_score-{score:.4f}.png'
                search_text.render_chip(row,dataset.imagery,rank,score,path,budget);pngs.append(path)
                results.append({'rank':rank,'score':score,'shard_id':shard,**{name:row[name] for name in fields},
                    'source_uri':dataset.qualify(row['source_uri']),'dataset_source_uri':row['source_uri'],'dataset_id':dataset.dataset_id,
                    'release_year':dataset.release_year,'acquisition_date':dataset.acquisition_date(row['source_uri']),'png':path.name})
            search_text.write_contact_sheet(pngs,destination/'contact-sheet.png',budget)
            budget.atomic_json(destination/'results.json',{'query':args.query,'run_fingerprint':root['run_fingerprint'],
                'collection_dir':str(datasets[0].directory),'collection_fingerprints':{dataset.dataset_id:recipe['run_fingerprint'] for dataset,recipe,_ in entries},
                'scope':'base-only' if base_only else 'base-plus-gapfill','collection_state':entries[0][2]['state'],
                'partial_collection':any(catalog['state']!='complete' for _,_,catalog in entries),
                'searched_source_tiles':sum(len(catalog['shards']) for _,_,catalog in entries),
                'total_source_tiles':sum(catalog['total_sources'] for _,_,catalog in entries),
                'searched_chip_count':sum(catalog['chip_count'] for _,_,catalog in entries),'search':details,'results':results})
            print(f"Retrieval: {details['backend']} in {details['search_seconds']:.2f}s\nResults: {destination}",flush=True)
            return 0
        finally:
            model=None;gc.collect()
            torch=sys.modules.get('torch')
            if torch is not None and torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=='__main__':raise SystemExit(main())
