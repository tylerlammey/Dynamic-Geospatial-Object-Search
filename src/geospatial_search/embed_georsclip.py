"""Stream NYS GeoTIFF chips into resumable, atomic GeoRSCLIP embedding bundles."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import islice
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

import numpy as np
import rasterio
from tqdm import tqdm

from .bundles import discard_staging, publish_bundle, validate_bundle
from .download_nys_naip import Budget, PauseRequested, job_lock, sha256
from .embedding_control import gpu_access
from .model import encode_images, fingerprint, load_config, load_model
from .paths import CONFIG, EMBEDDINGS, JOB, contained
from .tiling import Footprints, Grid, Source, load_inventory, read_rgb, summarize, validate_source


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def duration(seconds):
    if seconds is None:
        return 'estimating'
    seconds = max(0,int(seconds))
    hours,remainder = divmod(seconds,3600)
    minutes,seconds = divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def progress_metrics(total, processed, elapsed, session_elapsed, rate, state):
    if total < 0 or not 0 <= processed <= total:
        raise ValueError('Progress must be bounded by the candidate inventory')
    eta = 0.0 if state == 'complete' else (total-processed)/rate if rate and state == 'running' else None
    return {'percent_complete':100*processed/total if total else 100.0,
            'elapsed_seconds':max(0,elapsed),'elapsed_hms':duration(elapsed),
            'session_elapsed_seconds':max(0,session_elapsed),
            'candidates_per_second':rate,'eta_seconds':eta,'eta_hms':duration(eta)}


@dataclass
class Failure:
    error: BaseException


def put(queue, value, stop):
    while not stop.is_set():
        try:
            queue.put(value,timeout=.2)
            return
        except Full:
            pass


def prepare_source(source: Source, grid: Grid, config: dict, preprocess, queue: Queue,
                   stop: threading.Event, run_fingerprint: str, model_contract: dict, max_chips=None):
    """One reader per TIFF, bounded prefetch, CPU transforms only."""
    import torch
    try:
        before = (source.path.stat().st_size,source.path.stat().st_mtime_ns)
        if sha256(source.path) != source.sha256:
            raise ValueError(f'Source checksum failed: {source.uri}')
        scanned = skipped = 0
        images,rows = [],[]
        with rasterio.Env(GDAL_PAM_ENABLED=False):
            with rasterio.open(source.path) as dataset:
                validate_source(dataset,source)
                footprints = Footprints(dataset,grid.size)
                windows = grid.windows(source.width,source.height)
                if max_chips:
                    windows = islice(windows,max_chips)
                for window in windows:
                    if stop.is_set():
                        return
                    scanned += 1
                    mask = dataset.dataset_mask(window=window)
                    fraction = float(np.count_nonzero(mask)/mask.size)
                    if fraction < grid.min_valid_fraction:
                        skipped += 1
                        continue
                    images.append(preprocess(read_rgb(dataset,window)))
                    rows.append({'dataset_id':config['dataset_id'],
                        'chip_id':f"{config['dataset_id']}/{source.uri}/{grid.scale_id}/x{int(window.col_off):07d}_y{int(window.row_off):07d}",
                        'source_uri':source.uri,'source_tile_id':source.path.stem,'source_sha256':source.sha256,
                        'pixel_x':int(window.col_off),'pixel_y':int(window.row_off),
                        'pixel_width':grid.size,'pixel_height':grid.size,'valid_fraction':fraction,
                        'scale_id':grid.scale_id,'rgb_band_order':'1,2,3','model_id':config['model_id'],
                        'model_fingerprint':model_contract['model_fingerprint'],
                        'preprocess_fingerprint':model_contract['preprocess_fingerprint'],
                        'run_fingerprint':run_fingerprint,**footprints.metadata(window)})
                    if len(images) == config['batch_size']:
                        inputs = torch.stack(images)
                        if torch.cuda.is_available() and not torch.version.hip:
                            inputs = inputs.pin_memory()
                        put(queue,('batch',inputs,rows,scanned),stop)
                        images,rows = [],[]
                if images:
                    inputs = torch.stack(images)
                    if torch.cuda.is_available() and not torch.version.hip:
                        inputs = inputs.pin_memory()
                    put(queue,('batch',inputs,rows,scanned),stop)
        if before != (source.path.stat().st_size,source.path.stat().st_mtime_ns):
            raise ValueError(f'Source changed during embedding: {source.uri}')
        put(queue,('done',scanned,skipped),stop)
    except BaseException as exc:
        put(queue,Failure(exc),stop)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--job-root',type=Path,default=JOB)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--plan',action='store_true',help='Save inventory/count estimates; no GPU inference')
    parser.add_argument('--smoke-test',action='store_true',help='128 candidate chips from one TIFF per UTM zone')
    parser.add_argument('--resume',action='store_true',help='Verify completed compatible bundles and resume')
    parser.add_argument('--model-cache-dir',type=Path)
    parser.add_argument('--offline',action='store_true',help='Require the pinned checkpoint in the existing cache')
    parser.add_argument('--disable-amp',action='store_true')
    return parser.parse_args(argv)


def status_write(budget, output, payload):
    try:
        budget.atomic_json(output/'status.json',{'updated_utc':utc_now(),**payload})
    except PermissionError:
        pass  # Display locks cannot discard durable embedding bundles.


def build_recipe(config, sources, manifest_hash, grid, model_contract, smoke):
    compatible_model = {key:model_contract[key] for key in (
        'model_id','model_filename','model_revision','arch','base_pretrained','model_fingerprint',
        'preprocess_fingerprint','embedding_dim','embedding_dtype','metric','amp')}
    dependencies = {name:version(name) for name in ('numpy','rasterio','pyproj','pillow','pyarrow','open-clip-torch','torch')}
    code_hashes = {name:sha256(Path(__file__).with_name(name)) for name in
                   ('embed_georsclip.py','tiling.py','model.py','bundles.py')}
    return {'format_version':1,'dataset_id':config['dataset_id'],'source_manifest_sha256':manifest_hash,
            'chip_size':grid.size,'stride':grid.stride,'min_valid_fraction':grid.min_valid_fraction,
            'edge_policy':'full-size final anchored window; no padding','geometry_crs':'EPSG:4326',
            'mask_policy':'source validity mask; keep native RGB in accepted windows',
            'rgb_band_order':[1,2,3],'scale_id':grid.scale_id,'model':compatible_model,
            'dependencies':dependencies,'code_sha256':code_hashes,'scope':'smoke' if smoke else 'statewide',
            'max_candidate_chips_per_tile':128 if smoke else None,
            'sources':[{'uri':source.uri,'sha256':source.sha256} for source in sources]}


def main(argv=None):
    session_started = time.monotonic()
    args = parse_args(argv)
    config = load_config(args.config)
    for name in ('batch_size','workers'):
        if getattr(args,name) is not None:
            if getattr(args,name) <= 0:
                raise ValueError(f'{name} must be positive')
            config[name] = getattr(args,name)
    if args.disable_amp:
        config['amp'] = False
    if config['workers'] > 8:
        raise ValueError('At most eight prefetched source workers are supported')
    grid = Grid(config['chip_size'],config['stride'],config['min_valid_fraction'])
    job_root = args.job_root.resolve()
    sources,manifest_hash = load_inventory(job_root)
    print(json.dumps(summarize(sources,grid),indent=2),flush=True)
    if args.plan:
        budget = Budget(job_root)
        plan = {'created_utc':utc_now(),'source_manifest_sha256':manifest_hash,
                'config':config,**summarize(sources,grid)}
        destination = job_root/'tiling'/grid.scale_id/'plan.json'
        budget.atomic_json(destination,plan)
        print(f'Plan saved: {destination}',flush=True)
        return 0
    if args.smoke_test:
        by_crs = {}
        for source in sources:
            by_crs.setdefault(source.crs,source)
        sources = list(by_crs.values())
        if args.output_dir is None:
            args.output_dir = job_root/'embeddings'/f'smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    if args.output_dir is None:
        args.output_dir = job_root/'embeddings'/f"georsclip-vith14-s{grid.size}-st{grid.stride}"
    output = contained(job_root,args.output_dir)
    for forbidden in ('imagery','metadata','records','temporary'):
        protected = job_root/forbidden
        if output == protected or output.is_relative_to(protected):
            raise ValueError('Embedding output must not overlap downloader source folders')
    if output.exists() and any(p.name not in ('worker.lock','runtime') for p in output.iterdir()) and not args.resume:
        raise FileExistsError('Output already exists; use --resume or a distinct output directory')
    output.mkdir(parents=True,exist_ok=True)
    with job_lock(output), gpu_access(job_root):
        budget = Budget(job_root)
        previous_status = json.loads((output/'status.json').read_text(encoding='utf-8')) if args.resume and (output/'status.json').exists() else {}
        elapsed_before = float(previous_status.get('elapsed_seconds',0))
        stage_seconds = {name:float(previous_status.get(name,0)) for name in
                         ('reader_wait_seconds','inference_seconds','publication_seconds')}
        status_write(budget,output,{**previous_status,'state':'loading',
            'pid':__import__('os').getpid(),'error':None,'eta_seconds':None,
            'eta_hms':'estimating','folder_bytes':budget.total,'folder_cap_bytes':budget.cap})
        if args.resume:
            for signal in ('STOP','PAUSE'):
                if (output/signal).exists():
                    budget.remove(output/signal)
        import torch
        torch.set_num_threads(1)
        print('Loading the pinned GeoRSCLIP checkpoint...',flush=True)
        model,preprocess,device,model_contract = load_model(config,cache_dir=args.model_cache_dir,offline=args.offline)
        recipe = build_recipe(config,sources,manifest_hash,grid,model_contract,args.smoke_test)
        run_fingerprint = fingerprint(recipe)
        recipe_path = output/'run.json'
        if recipe_path.exists():
            existing = json.loads(recipe_path.read_text(encoding='utf-8'))
            if existing.get('run_fingerprint') != run_fingerprint:
                raise ValueError('Different source/grid/model/preprocessing/code contract; choose a new output directory')
        else:
            budget.atomic_json(recipe_path,{'run_fingerprint':run_fingerprint,'recipe':recipe,
                                          'model_contract':model_contract,'created_utc':utc_now()})
        shards = output/'shards'
        shards.mkdir(exist_ok=True)
        discard_staging(budget,shards)
        completed = {}
        for source in sources:
            directory = shards/source.shard_id
            if directory.exists():
                run = validate_bundle(directory,run_fingerprint,verify_hashes=True)
                if run['source_uri'] != source.uri or run['source_sha256'] != source.sha256:
                    raise ValueError('Completed shard source identity differs')
                completed[source.uri] = run
        limit = 128 if args.smoke_test else None
        candidates = {source.uri:min(grid.count(source.width,source.height),limit) if limit else grid.count(source.width,source.height) for source in sources}
        total_candidates = sum(candidates.values())
        scanned = sum(run['scanned_chips'] for run in completed.values())
        embedded = sum(run['chip_count'] for run in completed.values())
        skipped = sum(run['skipped_chips'] for run in completed.values())
        stop = threading.Event()
        current = None
        rate_samples = []
        last_status = 0
        loop_started = time.monotonic()
        initial_scanned = scanned
        progress = None

        def checkpoint(state, force=False, error=None):
            nonlocal last_status
            tick = time.monotonic()
            pause_requested = (output/'PAUSE').exists()
            if state == 'running' and pause_requested:
                state = 'pausing'
            if force or tick-last_status >= 2:
                last_status = tick
                measured = rate_samples[-20:]
                loop_elapsed = tick-loop_started
                rate = (sum(count for count,seconds in measured)/sum(seconds for count,seconds in measured) if measured
                        else (scanned-initial_scanned)/loop_elapsed if scanned>initial_scanned and loop_elapsed>0 else None)
                metrics = progress_metrics(total_candidates,scanned,elapsed_before+tick-session_started,
                                           tick-session_started,rate,state)
                if progress is not None:
                    progress.set_postfix_str(f"runtime {metrics['elapsed_hms']} | ETA {metrics['eta_hms']} | "
                        f"vectors {embedded:,} | tiles {len(completed)}/{len(sources)}",refresh=True)
                status_write(budget,output,{'state':state,'pid':__import__('os').getpid(),
                    'current_tile':current,'completed_sources':len(completed),'total_sources':len(sources),
                    'processed_candidate_chips':scanned,'total_candidate_chips':total_candidates,
                    'embedded_chips':embedded,'skipped_chips':skipped,
                    'durable_embedded_chips':sum(run['chip_count'] for run in completed.values()),
                    'pause_requested':pause_requested,
                    **metrics,**stage_seconds,
                    'batch_size':config['batch_size'],'workers':config['workers'],
                    'gpu':model_contract['gpu'],'error':error,'folder_bytes':budget.total,
                    'folder_cap_bytes':budget.cap})

        def save_collection(state):
            entries = [{'shard_id':source.shard_id,'source_uri':source.uri,'chip_count':completed[source.uri]['chip_count']}
                       for source in sources if source.uri in completed]
            budget.atomic_json(output/'collection.json',{'format_version':1,'run_fingerprint':run_fingerprint,
                'state':state,'scope':recipe['scope'],'total_sources':len(sources),'completed_sources':len(completed),
                'chip_count':sum(e['chip_count'] for e in entries),'embedding_dim':model_contract['embedding_dim'],
                'geometry_crs':'EPSG:4326','shards':entries})

        pending = [source for source in sources if source.uri not in completed]
        progress = tqdm(total=total_candidates,initial=scanned,desc='NYS embeddings',unit='window',dynamic_ncols=True,
                        mininterval=2,bar_format='{desc}: {percentage:6.2f}% |{bar}| {n_fmt}/{total_fmt}{postfix}')
        checkpoint('running',True)
        save_collection('running')
        executor = ThreadPoolExecutor(max_workers=config['workers'],thread_name_prefix='nys-reader')
        active = []
        pending_iterator = iter(pending)

        def submit_next():
            source = next(pending_iterator,None)
            if source:
                queue = Queue(maxsize=config['prefetch_batches'])
                future = executor.submit(prepare_source,source,grid,config,preprocess,queue,stop,
                                         run_fingerprint,model_contract,limit)
                active.append((source,queue,future))

        try:
            for _ in range(config['workers']):
                submit_next()
            while active:
                if (output/'PAUSE').exists():
                    raise PauseRequested('Graceful pause: all computed chips of completed source tiles are saved')
                source,queue,future = active.pop(0)
                current = source.uri
                started = time.monotonic()
                capacity = candidates[source.uri]
                if capacity > 50_000:
                    raise ValueError('Source requires over 50,000 chips; split it before using this pipeline')
                features = np.empty((capacity,model_contract['embedding_dim']),dtype=np.float16)
                rows = []
                seen = 0
                while True:
                    if (output/'STOP').exists():
                        raise PauseRequested('STOP requested; completed source bundles are preserved')
                    wait_started = time.monotonic()
                    try:
                        item = queue.get(timeout=.2)
                    except Empty:
                        if future.done():
                            future.result()
                            raise RuntimeError('Reader ended without completion marker')
                        checkpoint('running')
                        continue
                    finally:
                        stage_seconds['reader_wait_seconds'] += time.monotonic()-wait_started
                    if isinstance(item,Failure):
                        raise item.error
                    if item[0] == 'done':
                        _,source_scanned,source_skipped = item
                        future.result()
                        scanned += source_scanned-seen
                        progress.update(source_scanned-seen)
                        skipped += source_skipped
                        break
                    _,inputs,batch_rows,batch_scanned = item
                    inference_started = time.monotonic()
                    try:
                        vectors = encode_images(model,inputs,device,config['amp'])
                    finally:
                        stage_seconds['inference_seconds'] += time.monotonic()-inference_started
                    offset = len(rows)
                    features[offset:offset+len(batch_rows)] = vectors.astype(np.float16)
                    for index,row in enumerate(batch_rows,offset):
                        row['embedding_id'] = index
                    rows.extend(batch_rows)
                    scanned += batch_scanned-seen
                    progress.update(batch_scanned-seen)
                    seen = batch_scanned
                    embedded += len(batch_rows)
                    checkpoint('running')
                run = {**model_contract,'run_fingerprint':run_fingerprint,'dataset_id':config['dataset_id'],
                    'source_uri':source.uri,'source_sha256':source.sha256,'source_crs_wkt':rows[0]['crs_wkt'] if rows else source.crs,
                    'geometry_crs':'EPSG:4326','chip_size':grid.size,'stride':grid.stride,
                    'scale_id':grid.scale_id,'min_valid_fraction':grid.min_valid_fraction,
                    'scanned_chips':source_scanned,'skipped_chips':source_skipped,'created_at':utc_now()}
                publication_started = time.monotonic()
                completed[source.uri] = publish_bundle(budget,shards/source.shard_id,features[:len(rows)],rows,run)
                save_collection('running')
                stage_seconds['publication_seconds'] += time.monotonic()-publication_started
                rate_samples.append((source_scanned,time.monotonic()-started))
                checkpoint('running',True)
                if not (output/'PAUSE').exists():
                    submit_next()
                if (output/'PAUSE').exists() and len(completed)<len(sources):
                    raise PauseRequested('Graceful pause: current source tile saved; resume will skip completed tiles')
            save_collection('complete')
            budget.atomic_json(output/'completion.json',{'completed_utc':utc_now(),'run_fingerprint':run_fingerprint,
                'source_tiles':len(completed),'chip_count':embedded,'skipped_chips':skipped,
                'elapsed_seconds':elapsed_before+time.monotonic()-session_started,**stage_seconds})
            current = None
            checkpoint('complete',True)
            print(f'Complete: {embedded:,} vectors. Collection: {output}',flush=True)
            return 0
        except BaseException as exc:
            state = 'paused' if isinstance(exc,(PauseRequested,KeyboardInterrupt)) else 'failed'
            stop.set()
            # Report the durable checkpoint, excluding an interrupted source.
            scanned = sum(run['scanned_chips'] for run in completed.values())
            embedded = sum(run['chip_count'] for run in completed.values())
            skipped = sum(run['skipped_chips'] for run in completed.values())
            progress.n = scanned
            save_collection(state)
            checkpoint(state,True,str(exc))
            if state == 'paused':
                print(f'Paused safely: {embedded:,} saved vectors in {len(completed)} source tiles. Resume with -Resume.',flush=True)
                return 0
            raise
        finally:
            stop.set()
            executor.shutdown(wait=True,cancel_futures=True)
            progress.close()
            # Release inference weights and cached buffers before unlocking the
            # collection so a subsequent text query can safely use this GPU.
            del model
            import gc
            gc.collect()  # OpenCLIP modules can retain cyclic parameter references.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
