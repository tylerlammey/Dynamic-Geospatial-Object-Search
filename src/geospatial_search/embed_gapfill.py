"""Supplement-only adapter around the unchanged durable embedding engine."""
from itertools import islice
import json
from pathlib import Path
import threading

import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Point,shape
from shapely.ops import transform,unary_union

from . import embed_georsclip as engine
from .download_nys_naip import Budget,sha256
from .gapfill import DATASET,directory,load_inventory
from .model import load_config
from .paths import REPOSITORY
from .tiling import Footprints,Grid,read_rgb,summarize,validate_source

CONFIG=REPOSITORY/'configs/nys-georsclip-gapfill.json'
CONTEXT_METERS=76.8


def selection_geometry(manifest):
    projector=Transformer.from_crs(4326,26918,always_xy=True)
    return unary_union([transform(projector.transform,shape(gap['geometry'])) for gap in manifest['gaps']]).buffer(CONTEXT_METERS)


def prepare_source(selection,source,grid,config,preprocess,queue,stop,run_fingerprint,model_contract,max_chips=None):
    """Keep full native chips near gaps, independently of the 95% mask rule."""
    import torch
    try:
        before=(source.path.stat().st_size,source.path.stat().st_mtime_ns)
        if sha256(source.path)!=source.sha256:raise ValueError('Gap-fill source checksum failed')
        scanned=skipped=0;images=[];rows=[]
        with rasterio.Env(GDAL_PAM_ENABLED=False):
            with rasterio.open(source.path) as ds:
                validate_source(ds,source);footprints=Footprints(ds,grid.size)
                windows=grid.windows(source.width,source.height)
                if max_chips:windows=islice(windows,max_chips)
                def flush():
                    inputs=torch.stack(images)
                    if torch.cuda.is_available() and not torch.version.hip:inputs=inputs.pin_memory()
                    engine.put(queue,('batch',inputs,list(rows),scanned),stop)
                    images.clear();rows.clear()
                for window in windows:
                    if stop.is_set():return
                    scanned+=1
                    center=ds.transform*(window.col_off+grid.size/2,window.row_off+grid.size/2)
                    if not selection.covers(Point(*center)):
                        skipped+=1;continue
                    mask=ds.dataset_mask(window=window);fraction=float(np.count_nonzero(mask)/mask.size)
                    if fraction<grid.min_valid_fraction:
                        skipped+=1;continue
                    images.append(preprocess(read_rgb(ds,window)))
                    rows.append({'dataset_id':config['dataset_id'],
                        'chip_id':f"{config['dataset_id']}/{source.uri}/{grid.scale_id}/x{int(window.col_off):07d}_y{int(window.row_off):07d}",
                        'source_uri':source.uri,'source_tile_id':source.path.stem,'source_sha256':source.sha256,
                        'pixel_x':int(window.col_off),'pixel_y':int(window.row_off),'pixel_width':grid.size,'pixel_height':grid.size,
                        'valid_fraction':fraction,'scale_id':grid.scale_id,'rgb_band_order':'1,2,3','model_id':config['model_id'],
                        'model_fingerprint':model_contract['model_fingerprint'],'preprocess_fingerprint':model_contract['preprocess_fingerprint'],
                        'run_fingerprint':run_fingerprint,**footprints.metadata(window)})
                    if len(images)==config['batch_size']:flush()
                if images:flush()
        if before!=(source.path.stat().st_size,source.path.stat().st_mtime_ns):raise ValueError('Gap-fill source changed during embedding')
        engine.put(queue,('done',scanned,skipped),stop)
    except BaseException as exc:engine.put(queue,engine.Failure(exc),stop)


def main(argv=None):
    import sys
    arguments=list(sys.argv[1:] if argv is None else argv)
    args=engine.parse_args(arguments);job=args.job_root.resolve();root=directory(job)
    if args.config==engine.CONFIG:arguments += ['--config',str(CONFIG)]
    config=load_config(args.config if args.config!=engine.CONFIG else CONFIG)
    if config['dataset_id']!=DATASET:raise ValueError('Use the separate gap-fill model configuration')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    sources,digest=load_inventory(job)
    selection=selection_geometry(manifest)
    if args.plan:
        budget=Budget(job);plan={'dataset_id':DATASET,'source_manifest_sha256':digest,
            'selection':'centroid inside target gaps buffered by 76.8 meters',**summarize(sources,Grid())}
        budget.atomic_json(root/'tiling/native060cm-s256-st192/plan.json',plan)
        print(json.dumps(plan,indent=2));return 0
    if args.output_dir is None and not args.smoke_test:
        arguments += ['--output-dir',str(root/'embeddings/georsclip-vith14-s256-st192')]
    elif args.output_dir is not None and not args.output_dir.resolve().is_relative_to(root/'embeddings'):
        raise ValueError('Gap-fill embeddings belong in the supplement embeddings folder')
    if args.smoke_test and args.output_dir is None:
        from datetime import datetime,timezone
        arguments += ['--output-dir',str(root/'embeddings'/f'smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}')]
    original_inventory=engine.load_inventory;original_recipe=engine.build_recipe;original_reader=engine.prepare_source
    def recipe(*values):
        result=original_recipe(*values)
        result['scope']='smoke' if values[-1] else 'supplement'
        result['supplement']={'release_year':2021,'selection':'gap-centroid-with-native-context','context_meters':CONTEXT_METERS}
        result['code_sha256'].update({name:sha256(Path(__file__).with_name(name)) for name in ('embed_gapfill.py','gapfill.py')})
        return result
    # These bindings are scoped to this dedicated CLI process and restored on
    # exit. Base files/hashes stay unchanged; publication/pause/GPU locks/budget
    # remain implemented by the existing, tested embedding engine.
    engine.load_inventory=lambda selected_job:load_inventory(selected_job)
    engine.build_recipe=recipe
    engine.prepare_source=lambda *values:prepare_source(selection,*values)
    try:return engine.main(arguments)
    finally:
        engine.load_inventory=original_inventory;engine.build_recipe=original_recipe;engine.prepare_source=original_reader


if __name__=='__main__':raise SystemExit(main())
