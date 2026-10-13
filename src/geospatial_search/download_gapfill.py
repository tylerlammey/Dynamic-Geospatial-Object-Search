"""Resumable, budgeted acquisition of the verified 75 NYS gap-fill tiles."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import json
import math
import hashlib
import os
from pathlib import Path
import re
import threading
import time
import urllib.request

import numpy as np
from .download_nys_naip import (Budget,BudgetExceeded,PauseRequested,job_lock,sha256,now,verified_record,
    CHUNK,QUALITIES,check_pause,encode,save_memory,validate_source,windows)
from .gapfill import DATASET,REMOTE,ScopedBudget,directory
from .paths import JOB


def download(url,path,budget,max_bytes,progress=None,check_stop=True):
    """Separate 2021 allowlist; do not relax the original 2022 downloader."""
    if not url.startswith(REMOTE) or not re.fullmatch(r'(?:index\.html|m_\d{7}_[ns][ew]_18_060_\d{8}\.tif)',url[len(REMOTE):]):
        raise ValueError('Unexpected gap-fill source URL')
    path=budget.contained(path)
    if path.exists():budget.remove(path)
    path.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();size=0
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'NYS-gap-fill-download/1.0'}),timeout=90) as response:
        if response.geturl()!=url:raise ValueError('Unexpected provider redirect')
        expected=int(response.headers['Content-Length'])
        if expected>max_bytes:raise BudgetExceeded('Gap-fill source exceeds its reserved transfer allowance')
        with path.open('wb') as handle:
            while block:=response.read(CHUNK):
                if check_stop:check_pause(budget.root)
                size+=len(block)
                if size>max_bytes:raise BudgetExceeded('Gap-fill stream exceeded its reserved transfer allowance')
                with budget.lock:
                    budget.check(path,size);handle.write(block);handle.flush();budget.account(path,size)
                digest.update(block)
                if progress:progress('downloading',size,expected)
            os.fsync(handle.fileno())
        if size!=expected:raise IOError('Incomplete gap-fill source download')
        return {'bytes':size,'sha256':digest.hexdigest(),'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified')}


def process_tile(entry,budget,progress):
    from rasterio.warp import transform_bounds
    import rasterio
    existing=verified_record(entry,budget)
    if existing:return existing
    final=budget.root/'imagery/utm18'/entry['name'];record_path=budget.root/'records'/f"{entry['name']}.json"
    source_path=budget.root/'temporary'/f"{entry['name']}.source.part"
    output_path=budget.root/'temporary'/f"{entry['name']}.compressed.part"
    for path in (source_path,output_path):
        if path.exists():budget.remove(path)
    start=time.monotonic()
    info=download(entry['url'],source_path,budget,entry['http']['bytes'],progress)
    if info['etag']!=entry['http']['etag']:raise ValueError('Candidate source changed since the audited HTTP metadata')
    with rasterio.Env(GDAL_PAM_ENABLED=False,GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(source_path) as source:
            validate_source(source,entry)
            for quality in QUALITIES:
                memory=encode(source,entry,quality,budget.root,progress)
                if memory is None:continue
                try:output_hash=save_memory(memory,output_path,budget,progress)
                finally:memory.close()
                with rasterio.open(output_path) as written:
                    for window in windows(written.width,written.height):
                        check_pause(budget.root);written.read((1,2,3),window=window)
                if sha256(output_path)!=output_hash:raise ValueError('Published gap-fill RGB checksum differs')
                record={'source_url':entry['url'],'source_download':info,'output_sha256':output_hash,
                    'output_bytes':output_path.stat().st_size,'raw_rgb_bytes':entry['raw_rgb_bytes'],'jpeg_quality':quality,
                    'resolution_m_per_pixel':.6,'width':source.width,'height':source.height,'crs':source.crs.to_string(),
                    'transform':list(source.transform)[:6],'bounds_wgs84':list(transform_bounds(source.crs,'EPSG:4326',*source.bounds)),
                    'output':final.relative_to(budget.root).as_posix(),'verified_utc':now(),'elapsed_seconds':time.monotonic()-start,
                    'dataset_id':DATASET,'release_year':2021,'acquisition_date':entry['acquisition_date'],
                    'validity_mask':'internal; checked equal to source at every pixel',
                    'attribution':'Courtesy USDA-FSA; distributed by NOAA Digital Coast'}
                final.parent.mkdir(parents=True,exist_ok=True);budget.replace(output_path,final);budget.atomic_json(record_path,record)
                break
            else:raise BudgetExceeded('Gap-fill tile cannot meet the quality-75 output ceiling')
    budget.remove(source_path);return record


def prepare(job,budget):
    root=directory(job);root.mkdir(parents=True,exist_ok=True)
    audit_path=job/'verification/gap-patch-audit-20261012/verification.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if (audit['existing_tile_count']!=3805 or audit['missing_existing_downloads']
        or not audit['candidate_bounds_cover_holes'] or len(audit['replacement_tiles'])!=75):
        raise ValueError('Gap audit does not establish the intended 75-tile supplement')
    index=root/'metadata/index.html'
    download(REMOTE+'index.html',index,budget,2_000_000,check_stop=False)
    expected=(job/'verification/gap-patch-audit-20261012/noaa-2021-index.html').read_bytes()
    if index.read_bytes()!=expected:raise ValueError('Provider index changed since the audit; re-audit before downloading')
    tiles=[]
    html=index.read_text(encoding='utf-8')
    for candidate in audit['replacement_tiles']:
        name=candidate['name']
        if not re.fullmatch(r'm_\d{7}_[ns][ew]_18_060_\d{8}\.tif',name) or candidate['url']!=REMOTE+name or candidate['url'] not in html:
            raise ValueError('Candidate is not in the allowlisted provider inventory')
        if candidate['crs']!='EPSG:26918' or not np.allclose([candidate['transform'][0],candidate['transform'][4]],[.6,-.6],rtol=0,atol=1e-9):
            raise ValueError('Candidate native grid differs')
        raw=candidate['width']*candidate['height']*3
        date=candidate['acquisition_date_from_filename']
        tiles.append({**candidate,'zone':'18','source_dataset':'NY_NAIP_2021_9664','raw_rgb_bytes':raw,
            'listed_bytes':candidate['http']['bytes'],'max_output_bytes':math.ceil(raw/8)+1024**2,
            'dataset_id':DATASET,'release_year':2021,'acquisition_date':f'{date[:4]}-{date[4:6]}-{date[6:]}'})
    if len({t['name'] for t in tiles})!=75:raise ValueError('Duplicate candidate')
    manifest={'format_version':1,'dataset_id':DATASET,'release_year':2021,'tile_count':75,
        'bands':[1,2,3],'resolution_m_per_pixel':.6,'source_index_url':REMOTE+'index.html',
        'metadata_sha256':{'index.html':sha256(index)},'audit_sha256':sha256(audit_path),
        'gaps':audit['gaps'],'network_bytes':sum(t['http']['bytes'] for t in tiles),'tiles':tiles}
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Existing supplement manifest differs; preserve it and investigate')
    if not path.exists():budget.atomic_json(path,manifest)
    return manifest


def pilot_tiles(manifest):
    chosen=[]
    for gap in manifest['gaps']:
        names={c['name'] for c in gap['candidates']}
        candidates=[t for t in manifest['tiles'] if t['name'] in names]
        center=gap['center']
        nearest=min(candidates,key=lambda t:(t['nominal_center'][0]-center[0])**2+(t['nominal_center'][1]-center[1])**2)
        boundary=max(candidates,key=lambda t:abs(t['nominal_center'][1]-center[1]))
        chosen.extend([nearest,boundary])
    return {t['name'] for t in chosen}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-root',type=Path,default=JOB)
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv);job=args.job_root.resolve();root=directory(job)
    if not 1<=args.workers<=4:raise ValueError('Use one to four workers')
    root.mkdir(parents=True,exist_ok=True)
    with job_lock(root):
        budget=Budget(job);scope=ScopedBudget(budget,root)
        if (job/'STOP').exists():raise PauseRequested('Primary download STOP signal is present')
        manifest=prepare(job,budget)
        if args.prepare:
            print(json.dumps({'tiles':75,'network_bytes':manifest['network_bytes'],'folder_bytes':budget.total,'root':str(root)},indent=2));return 0
        if (root/'STOP').exists():budget.remove(root/'STOP')
        selected=pilot_tiles(manifest) if args.pilot else {t['name'] for t in manifest['tiles']}
        chosen=[t for t in manifest['tiles'] if t['name'] in selected]
        completed={};active={};state_lock=threading.RLock();start=time.monotonic();last=[0.];downloaded=[0]
        def checkpoint(state='running',error=None,force=False):
            with state_lock:
                if not force and time.monotonic()-last[0]<1:return
                last[0]=time.monotonic();elapsed=last[0]-start
                remaining=len(chosen)-len(completed)
                eta=elapsed/len(completed)*remaining if completed and state=='running' else None
                scope.atomic_json(root/'status.json',{'state':state,'pid':__import__('os').getpid(),'updated_utc':now(),
                    'completed_tiles':len(completed),'total_tiles':len(chosen),'percent_complete':100*len(completed)/len(chosen),
                    'elapsed_seconds':elapsed,'eta_seconds':eta,'folder_bytes':budget.total,'folder_cap_bytes':budget.cap,
                    'compressed_imagery_bytes':sum(r['output_bytes'] for r in completed.values()),'network_bytes_this_session':downloaded[0],
                    'configured_workers':args.workers,'active_workers':list(active.values()),'error':error,'pilot':args.pilot})
        def progress_for(tile):
            def update(phase,processed,total,force=False):
                with state_lock:
                    active[tile['name']]={'tile':tile['name'],'phase':phase,'phase_percent':100*processed/total if total else 0,
                        'processed_bytes_or_pixels':processed,'total_bytes_or_pixels':total}
                checkpoint(force=force)
            return update
        def work(tile):
            record=process_tile(tile,scope,progress_for(tile))
            record.update(dataset_id=DATASET,release_year=2021,acquisition_date=tile['acquisition_date'])
            scope.atomic_json(root/'records'/f"{tile['name']}.json",record)
            return record
        try:
            checkpoint('verifying',force=True)
            pending=[]
            for tile in chosen:
                record=verified_record(tile,scope)
                if record:
                    if record.get('dataset_id')!=DATASET or record.get('release_year')!=2021:raise ValueError('Saved gap-fill provenance is unlabeled')
                    completed[tile['name']]=record
                else:pending.append(tile)
            checkpoint(force=True)
            with ThreadPoolExecutor(max_workers=args.workers,thread_name_prefix='gap-fill') as pool:
                iterator=iter(pending);running={}
                def submit():
                    tile=next(iterator,None)
                    if tile:running[pool.submit(work,tile)]=tile
                for _ in range(args.workers):submit()
                while running:
                    done,_=wait(running,timeout=1,return_when=FIRST_COMPLETED)
                    checkpoint()
                    for future in done:
                        tile=running.pop(future);record=future.result()
                        with state_lock:
                            completed[tile['name']]=record;active.pop(tile['name'],None)
                            downloaded[0]+=record['source_download']['bytes']
                        checkpoint(force=True);submit()
            if args.pilot:
                scope.atomic_json(root/'pilot.json',{'completed_utc':now(),'tiles':sorted(selected),'imagery_bytes':sum(r['output_bytes'] for r in completed.values())})
            else:
                scope.atomic_json(root/'completion.json',{'completed_utc':now(),'tile_count':75,'manifest_sha256':sha256(root/'manifest.json'),
                    'imagery_bytes':sum(r['output_bytes'] for r in completed.values()),'dataset_id':DATASET})
            checkpoint('complete',force=True);return 0
        except Exception as exc:
            checkpoint('paused' if isinstance(exc,PauseRequested) else 'failed',str(exc),force=True);raise


if __name__=='__main__':raise SystemExit(main())
