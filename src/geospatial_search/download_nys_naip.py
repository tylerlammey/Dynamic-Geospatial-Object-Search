"""Resumable, storage-bounded native-resolution New York NAIP conversion."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import threading
import uuid
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window
from rasterio.enums import ColorInterp
from rasterio.warp import transform_bounds
from affine import Affine

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'nys-naip-2022-rgb-060m'
BASE = 'https://coastalimagery.blob.core.windows.net/digitalcoast/NY_NAIP_2022_9986/'
CAP = 300_000_000_000
RESERVE = 5_000_000_000
CHUNK = 4 * 1024 * 1024
QUALITIES = (83, 80, 77, 75)
DEFAULT_WORKERS = 4


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return call


def now():
    return datetime.now(timezone.utc).isoformat()


class BudgetExceeded(RuntimeError):
    pass


class PauseRequested(RuntimeError):
    pass


class Budget:
    """Shared accounting; serialize each check/write/account transaction."""
    def __init__(self, root, cap=CAP, reserve=RESERVE):
        self.lock = threading.RLock()
        self.root = Path(root).resolve()
        self.cap = cap
        self.reserve = reserve
        self.sizes = {}
        for path in self.root.rglob('*'):
            if path.is_symlink():
                raise RuntimeError(f'Symlinks are not permitted in the job folder: {path}')
            if path.is_file():
                self.sizes[path.resolve()] = path.stat().st_size
        self.total = sum(self.sizes.values())
        self.peak = self.total
        previous_status = self.root / 'status.json'
        if previous_status.exists():
            prior_peak = json.loads(previous_status.read_text(encoding='utf-8')).get('peak_folder_bytes', 0)
            self.peak = max(self.peak, prior_peak)
        if self.total > cap - reserve:
            raise BudgetExceeded('Existing folder exceeds the working budget')

    def contained(self, path):
        path = Path(path).resolve()
        if path == self.root or not path.is_relative_to(self.root):
            raise ValueError(f'Path escapes the job folder: {path}')
        return path

    @synchronized
    def check(self, path, new_size):
        path = self.contained(path)
        if new_size < 0:
            raise ValueError('Negative file size')
        old = self.sizes.get(path, 0)
        growth = max(0, new_size - old)
        if self.total + growth > self.cap - self.reserve:
            raise BudgetExceeded(f'Write would exceed {self.cap:,}-byte folder cap')
        if growth and shutil.disk_usage(self.root).free < growth + min(self.reserve, 1_000_000_000):
            raise BudgetExceeded('Insufficient free disk space for the guarded write')
        return path

    @synchronized
    def account(self, path, size):
        path = self.contained(path)
        self.total += size - self.sizes.get(path, 0)
        self.sizes[path] = size
        self.peak = max(self.peak, self.total)

    @synchronized
    def write_bytes(self, path, content, append=False):
        path = self.contained(path)
        old = path.stat().st_size if append and path.exists() else 0
        self.check(path, old + len(content))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('ab' if append else 'wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        self.account(path, path.stat().st_size)

    @synchronized
    def atomic_json(self, path, data):
        path = self.contained(path)
        temp = path.with_name(path.name + '.new')
        self.write_bytes(temp, (json.dumps(data, indent=2) + '\n').encode('utf-8'))
        self.replace(temp, path)

    @synchronized
    def replace(self, source, destination):
        source, destination = self.contained(source), self.contained(destination)
        size = source.stat().st_size
        # Windows viewers, antivirus, and even brief status reads can hold a
        # handle without FILE_SHARE_DELETE. Retry without losing the old file.
        for attempt in range(20):
            try:
                os.replace(source, destination)
                break
            except PermissionError:
                if attempt==19:
                    raise
                time.sleep(min(.01*2**attempt,.1))
        self.account(source, 0)
        self.sizes.pop(source, None)
        self.account(destination, size)

    @synchronized
    def remove(self, path):
        path = self.contained(path)
        if path.exists():
            path.unlink()
        self.total -= self.sizes.pop(path, 0)


def check_pause(root):
    if (Path(root) / 'STOP').exists():
        raise PauseRequested('STOP file found; pausing cleanly')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, budget, max_bytes, *, check_stop=True, progress=None):
    """Stream one HTTP response through a pre-write size guard."""
    if not url.startswith(BASE):
        raise ValueError('Unexpected source host or dataset')
    path = budget.contained(path)
    if path.exists():
        budget.remove(path)
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={'User-Agent': 'NYS-NAIP-native-RGB-download/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        length = response.headers.get('Content-Length')
        expected = int(length) if length is not None else None
        if expected is not None and expected > max_bytes:
            raise BudgetExceeded('Source exceeds its reserved download allowance')
        size = 0
        if progress:
            progress('downloading', 0, expected or max_bytes, force=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            while block := response.read(CHUNK):
                if check_stop:
                    check_pause(budget.root)
                new_size = size + len(block)
                if new_size > max_bytes:
                    raise BudgetExceeded('Source stream exceeds its reserved allowance')
                with budget.lock:
                    budget.check(path, new_size)
                    handle.write(block)
                    handle.flush()
                    budget.account(path, new_size)
                digest.update(block)
                size = new_size
                if progress:
                    progress('downloading', size, expected or max_bytes)
            os.fsync(handle.fileno())
        if expected is not None and size != expected:
            raise IOError(f'Incomplete download: expected {expected}, received {size}')
        return {'bytes': size, 'sha256': digest.hexdigest(),
                'etag': response.headers.get('ETag'), 'last_modified': response.headers.get('Last-Modified')}


def prepare(budget):
    metadata = budget.root / 'metadata'
    for name in ('index.html', 'ny_naip_2022_17.vrt', 'ny_naip_2022_18.vrt', 'ny_naip_2022_19.vrt'):
        path = metadata / name
        if not path.exists():
            download(BASE + name, path, budget, 20_000_000)
    html = (metadata / 'index.html').read_text(encoding='utf-8')
    matches = re.findall(r'<a href="([^"\s]+\.tif)">[^<]+</a>\s*\(([0-9.]+)\s*(MB|GB)\)', html)
    urls = {}
    for url, size, unit in matches:
        if not url.startswith(BASE):
            raise ValueError('Unexpected imagery URL in provider index')
        name = url.rsplit('/', 1)[-1]
        if not re.fullmatch(r'm_\d{7}_(ne|nw|se|sw)_\d{2}_060_\d{8}\.tif', name):
            raise ValueError(f'Unexpected imagery filename or resolution: {name}')
        if name in urls:
            raise ValueError('Duplicate source filename')
        urls[name] = {'url': url, 'listed_bytes': round(float(size) * (1024**2 if unit == 'MB' else 1024**3))}
    entries = {}
    for zone in (17, 18, 19):
        vrt = ET.parse(metadata / f'ny_naip_2022_{zone}.vrt').getroot()
        grid = Affine.from_gdal(*[float(value) for value in vrt.findtext('GeoTransform').split(',')])
        if not math.isclose(grid.a, 0.6) or not math.isclose(grid.e, -0.6) or grid.b or grid.d:
            raise ValueError('Provider index does not use the expected 0.6-meter grid')
        srs = rasterio.crs.CRS.from_wkt(vrt.findtext('SRS'))
        if srs.to_epsg() != 26900 + zone:
            raise ValueError('Unexpected native CRS')
        for source in vrt.find('VRTRasterBand'):
            if source.find('SourceFilename') is None:
                continue
            name = Path(source.findtext('SourceFilename')).name
            properties = source.find('SourceProperties').attrib
            src, dst = source.find('SrcRect').attrib, source.find('DstRect').attrib
            width, height = int(properties['RasterXSize']), int(properties['RasterYSize'])
            if src['xSize'] != dst['xSize'] or src['ySize'] != dst['ySize']:
                raise ValueError(f'Index resamples a source tile: {name}')
            transform = grid * Affine.translation(float(dst['xOff']) - float(src['xOff']),
                                                  float(dst['yOff']) - float(src['yOff']))
            entry = dict(name=name, zone=zone, crs=srs.to_string(), width=width, height=height,
                         transform=list(transform)[:6], raw_rgb_bytes=width*height*3,
                         max_output_bytes=math.ceil(width*height*3/8)+1024**2, **urls[name])
            if name in entries and entries[name] != entry:
                raise ValueError('Conflicting provider records')
            entries[name] = entry
    if set(entries) != set(urls) or len(entries) != 3805:
        raise ValueError('Incomplete or changed provider manifest; investigate before downloading')
    manifest = {'dataset': 'NY_NAIP_2022_9986', 'source': BASE+'index.html',
                'created_utc': now(), 'resolution_m_per_pixel': 0.6, 'bands': [1,2,3],
                'folder_cap_bytes': CAP, 'reserve_bytes': RESERVE, 'jpeg_quality_candidates': list(QUALITIES),
                'tile_count': len(entries), 'total_pixels': sum(e['width']*e['height'] for e in entries.values()),
                'raw_rgb_bytes': sum(e['raw_rgb_bytes'] for e in entries.values()),
                'source_listed_bytes': sum(e['listed_bytes'] for e in entries.values()),
                'max_total_output_bytes': sum(e['max_output_bytes'] for e in entries.values()),
                'metadata_sha256': {p.name:sha256(p) for p in metadata.iterdir() if p.is_file()},
                'tiles': sorted(entries.values(), key=lambda e:e['name'])}
    if manifest['max_total_output_bytes'] + 4*(max(e['listed_bytes'] for e in entries.values())+2*1024**2) + RESERVE >= CAP:
        raise BudgetExceeded('Planned output ceiling does not fit under the folder cap')
    budget.atomic_json(budget.root/'manifest.json', manifest)
    return manifest


def windows(width, height, block=1024):
    for row in range(0,height,block):
        for col in range(0,width,block):
            yield Window(col,row,min(block,width-col),min(block,height-row))


def validate_source(dataset, entry):
    if dataset.shape != (entry['height'],entry['width']):
        raise ValueError('Source dimensions differ from the pinned provider index')
    if dataset.crs is None or dataset.crs.to_string() != entry['crs']:
        raise ValueError('Source CRS differs from provider index')
    if not dataset.transform.almost_equals(Affine(*entry['transform']),precision=1e-6):
        raise ValueError('Source transform differs from provider index')
    if not all(math.isclose(value,0.6,abs_tol=1e-9) for value in dataset.res):
        raise ValueError('Source does not have 0.6-meter pixels')
    if dataset.count != 4 or any(value!='uint8' for value in dataset.dtypes):
        raise ValueError('Unexpected source band count or sample type')
    if dataset.colorinterp[:3] != (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
        raise ValueError('Unexpected source RGB band order')


def encode(source, entry, quality, root, progress=None):
    """Encode in RAM so GDAL cannot bypass the disk-size guard."""
    memory = MemoryFile()
    try:
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_PAM_ENABLED=False, GDAL_CACHEMAX=128*1024*1024):
            with memory.open(driver='GTiff',width=source.width,height=source.height,count=3,dtype='uint8',
                             crs=source.crs,transform=source.transform,tiled=True,blockxsize=256,blockysize=256,
                             compress='JPEG',photometric='YCBCR',jpeg_quality=quality, BIGTIFF='NO') as target:
                target.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
                target.update_tags(SOURCE_URL=entry['url'],SOURCE_DATASET='NY_NAIP_2022_9986',
                                   SOURCE_ACQUISITION_DATE=entry['name'][-12:-4],JPEG_QUALITY=str(quality),
                                   ATTRIBUTION='Courtesy USDA-FSA; distributed by NOAA Digital Coast',
                                   RESOLUTION_M_PER_PIXEL='0.6',BANDS='RGB: source bands 1,2,3; NIR omitted')
                processed = 0
                if progress:
                    progress('compressing', 0, source.width*source.height, force=True)
                for window in windows(source.width,source.height):
                    check_pause(root)
                    target.write(source.read((1,2,3),window=window),window=window)
                    target.write_mask(source.dataset_mask(window=window),window=window)
                    processed += int(window.width*window.height)
                    if progress:
                        progress('compressing', processed, source.width*source.height)
        if len(memory) > entry['max_output_bytes']:
            memory.close()
            return None
        with memory.open() as target:
            if target.count != 3 or target.crs != source.crs or target.transform != source.transform or target.shape != source.shape:
                raise ValueError('Compressed output changed RGB dimensions or georeferencing')
            if target.compression.value!='JPEG' or target.overviews(1):
                raise ValueError('Unexpected output compression or overviews')
            processed = 0
            if progress:
                progress('validating_pixels_and_mask', 0, source.width*source.height, force=True)
            for window in windows(source.width,source.height):
                check_pause(root)
                if not np.array_equal(target.dataset_mask(window=window),source.dataset_mask(window=window)):
                    raise ValueError('Compressed output changed the source validity mask')
                processed += int(window.width*window.height)
                if progress:
                    progress('validating_pixels_and_mask', processed, source.width*source.height)
        memory.seek(0)
        return memory
    except BaseException:
        memory.close()
        raise


def save_memory(memory, path, budget, progress=None):
    path = budget.contained(path)
    budget.check(path,len(memory))
    digest=hashlib.sha256()
    copied = 0
    if progress:
        progress('writing_compressed_tile', 0, len(memory), force=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        while block:=memory.read(CHUNK):
            check_pause(budget.root)
            with budget.lock:
                budget.check(path,copied+len(block))
                handle.write(block)
                handle.flush()
                budget.account(path,copied+len(block))
            digest.update(block)
            copied += len(block)
            if progress:
                progress('writing_compressed_tile', copied, len(memory))
        os.fsync(handle.fileno())
    return digest.hexdigest()


def event(budget, kind, **details):
    content=(json.dumps({'utc':now(),'event':kind,**details})+'\n').encode('utf-8')
    path=budget.root/'events.jsonl'
    with budget.lock:
        if path.exists() and path.stat().st_size+len(content)>50_000_000:
            raise BudgetExceeded('Event log exceeded its 50 MB allocation')
        budget.write_bytes(path,content,append=True)
    print(content.decode('utf-8').rstrip(),flush=True)


def verified_record(entry, budget):
    record_path=budget.root/'records'/f"{entry['name']}.json"
    final=budget.root/'imagery'/f"utm{entry['zone']}"/entry['name']
    if not record_path.exists() or not final.exists():
        return None
    record=json.loads(record_path.read_text(encoding='utf-8'))
    if (record.get('source_url')!=entry['url'] or record.get('jpeg_quality') not in QUALITIES
        or record.get('resolution_m_per_pixel')!=0.6 or record.get('output_bytes')!=final.stat().st_size
        or final.stat().st_size>entry.get('max_output_bytes',float('inf'))
        or record.get('output_sha256')!=sha256(final)):
        raise ValueError(f'Existing output failed provenance/checksum validation: {final}')
    return record


def process_tile(entry,budget,progress=None):
    final=budget.root/'imagery'/f"utm{entry['zone']}"/entry['name']
    record_path=budget.root/'records'/f"{entry['name']}.json"
    existing=verified_record(entry,budget)
    if existing:
        return existing
    # An output without its atomic provenance is not marked complete; replace only after revalidation.
    temp_source=budget.root/'temporary'/(entry['name']+'.source.part')
    temp_output=budget.root/'temporary'/(entry['name']+'.compressed.part')
    for path in (temp_source,temp_output):
        if path.exists():
            budget.remove(path)
    started=time.monotonic()
    event(budget,'tile_started',tile=entry['name'])
    source_info=download(entry['url'],temp_source,budget,entry['listed_bytes']+2*1024**2,progress=progress)
    with rasterio.Env(GDAL_PAM_ENABLED=False, GDAL_CACHEMAX=128*1024*1024):
        with rasterio.open(temp_source) as source:
            validate_source(source,entry)
            for quality in QUALITIES:
                memory=encode(source,entry,quality,budget.root,progress=progress)
                if memory is None:
                    event(budget,'compression_retry',tile=entry['name'],quality=quality)
                    continue
                try:
                    output_hash=save_memory(memory,temp_output,budget,progress=progress)
                finally:
                    memory.close()
                # Decode the on-disk bytes as well before publication.
                with rasterio.open(temp_output) as written:
                    if written.count!=3 or written.crs!=source.crs or written.transform!=source.transform or written.shape!=source.shape:
                        raise ValueError('Written output differs from validated in-memory output')
                    processed = 0
                    if progress:
                        progress('verifying_written_tile', 0, written.width*written.height, force=True)
                    for window in windows(written.width,written.height):
                        check_pause(budget.root)
                        written.read((1,2,3),window=window)
                        processed += int(window.width*window.height)
                        if progress:
                            progress('verifying_written_tile', processed, written.width*written.height)
                size=temp_output.stat().st_size
                if sha256(temp_output)!=output_hash:
                    raise ValueError('Written output checksum differs from encoded bytes')
                record={'source_url':entry['url'],'source_download':source_info,'output_sha256':output_hash,
                        'output_bytes':size,'raw_rgb_bytes':entry['raw_rgb_bytes'],
                        'compression_ratio':entry['raw_rgb_bytes']/size,'jpeg_quality':quality,
                        'resolution_m_per_pixel':0.6,'width':source.width,'height':source.height,
                        'crs':source.crs.to_string(),'transform':list(source.transform)[:6],
                        'bounds_wgs84':list(transform_bounds(source.crs,'EPSG:4326',*source.bounds)),
                        'output':final.relative_to(budget.root).as_posix(),'verified_utc':now(),
                        'elapsed_seconds':round(time.monotonic()-started,3),
                        'validity_mask':'internal; checked equal to source at every pixel',
                        'attribution':'Courtesy USDA-FSA; distributed by NOAA Digital Coast'}
                final.parent.mkdir(parents=True,exist_ok=True)
                budget.replace(temp_output,final)
                budget.atomic_json(record_path,record)
                break
            else:
                raise BudgetExceeded(f"Tile cannot fit its reserved budget at JPEG quality >=75: {entry['name']}")
    budget.remove(temp_source)
    event(budget,'tile_completed',tile=entry['name'],output_bytes=record['output_bytes'],
          quality=record['jpeg_quality'],ratio=round(record['compression_ratio'],3))
    return record


@contextmanager
def job_lock(root):
    """Windows/POSIX OS lock, released automatically after a crash."""
    path=Path(root)/'worker.lock'
    handle=path.open('a+b')
    try:
        handle.seek(0)
        if path.stat().st_size==0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('Another worker is already running in this folder')
    try:
        yield
    finally:
        handle.close()


def estimate_progress(manifest, completed):
    """Estimate remaining time from the last 30 measured tile runtimes and pixel work."""
    ordered = sorted(completed.values(), key=lambda record:record.get('verified_utc',''))
    # Concurrent task durations overlap. Completion intervals measure actual throughput.
    epoch = next((record.get('throughput_epoch') for record in reversed(ordered)
                  if record.get('throughput_epoch')), None)
    measured = ([record for record in ordered if record.get('throughput_epoch')==epoch]
                if epoch else ordered)
    workers=measured[-1].get('execution_workers',1) if measured else 1
    if epoch and workers>1 and len(measured)>=3*workers:
        measured=measured[workers:]  # Exclude initial concurrent pipeline fill.
    recent = measured[-30:]
    recent = [record for record in recent if record.get('elapsed_seconds',0)>0 and record.get('raw_rgb_bytes',0)>0]
    remaining = max(0,manifest['raw_rgb_bytes']-sum(record['raw_rgb_bytes'] for record in completed.values()))
    eta = None
    if not remaining:
        eta = 0
    elif recent and (not epoch or len(measured)>=workers):
        eta = math.ceil(remaining*sum(record.get('throughput_seconds',record['elapsed_seconds']) for record in recent)/sum(record['raw_rgb_bytes'] for record in recent))
    return {'percent_complete':100*len(completed)/manifest['tile_count'],
            'eta_seconds':eta,'eta_sample_tiles':len(recent),
            'eta_method':('recent completion intervals in this worker session, weighted by remaining RGB pixels'
                          if epoch else 'last 30 verified tile runtimes, weighted by remaining RGB pixels')}


def write_status(budget,manifest,state,completed,current=None,error=None,**details):
    if time.monotonic()<getattr(budget,'_status_retry_after',0):
        return False
    estimate = estimate_progress(manifest,completed)
    if state not in ('running','complete'):
        estimate['eta_seconds'] = None
    eta = estimate['eta_seconds']
    payload={'state':state,'updated_utc':now(),'pid':os.getpid(),
        'completed_tiles':len(completed),'total_tiles':manifest['tile_count'],
        'compressed_imagery_bytes':sum(r['output_bytes'] for r in completed.values()),
        'folder_bytes':budget.total,'peak_folder_bytes':budget.peak,'folder_cap_bytes':CAP,
        'current_tile':current,'error':error,'resolution_m_per_pixel':0.6,'resolution_px_per_m':1/0.6,
        **estimate,'estimated_completion_utc':(datetime.now(timezone.utc)+timedelta(seconds=eta)).isoformat() if eta is not None else None,
        **details}
    try:
        budget.atomic_json(budget.root/'status.json',payload)
        return True
    except PermissionError:
        # Only the live display may be skipped. Imagery and provenance writes
        # remain mandatory; completed tiles are recovered from records on resume.
        tick=time.monotonic()
        budget._status_retry_after=tick+10
        if tick-getattr(budget,'_last_status_warning',-float('inf'))>=60:
            print('Status display is temporarily locked by Windows; download continues and will retry.',flush=True)
            budget._last_status_warning=tick
        return False


class Progress:
    def __init__(self,budget,manifest,completed):
        self.budget,self.manifest,self.completed=budget,manifest,completed
        self.current=None
        self.started_utc=now()
        self.last_update=-float('inf')
        self.lock=threading.RLock()
        self.active={}
        self.workers=1

    @synchronized
    def __call__(self,phase,done,total,force=False,state='running'):
        tick=time.monotonic()
        if not force and tick-self.last_update<2:
            return
        self.last_update=tick
        write_status(self.budget,self.manifest,state,self.completed,self.current,
                     phase=phase,phase_completed=done,phase_total=total,
                     phase_percent=min(100,100*done/total) if total else 0,session_started_utc=self.started_utc,
                     configured_workers=self.workers,active_workers=list(self.active.values()))

    def worker(self,name):
        def update(phase,done,total,force=False):
            with self.lock:
                self.active[name]={'tile':name,'phase':phase,'phase_percent':min(100,100*done/total) if total else 0}
                self.current=name
                self(phase,done,total,force=force)
        return update


def tile_with_retries(entry,budget,progress):
    update=progress.worker(entry['name'])
    for attempt in range(1,4):
        check_pause(budget.root)
        update('connecting',0,1,force=True)
        try:
            return process_tile(entry,budget,progress=update)
        except (BudgetExceeded,PauseRequested,ValueError):
            raise
        except Exception as exc:
            event(budget,'tile_error',tile=entry['name'],attempt=attempt,error=str(exc))
            if attempt==3:
                raise
            update('retrying_connection',0,5*attempt,force=True)
            for _ in range(5*attempt):
                check_pause(budget.root)
                time.sleep(1)


def run_tiles(tiles,budget,progress,workers):
    """At most workers active tiles; never enqueue the statewide collection."""
    epoch=uuid.uuid4().hex
    last_completion=time.monotonic()
    entries=iter(tiles)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='naip-tile') as executor:
        pending={}
        def submit_next():
            check_pause(budget.root)
            entry=next(entries,None)
            if entry is not None:
                pending[executor.submit(tile_with_retries,entry,budget,progress)]=entry
        try:
            for _ in range(workers):
                submit_next()
            while pending:
                ready,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in ready:
                    entry=pending.pop(future)
                    record=future.result()
                    tick=time.monotonic()
                    record.update(throughput_seconds=round(tick-last_completion,6),throughput_epoch=epoch,
                                  execution_workers=workers,verified_utc=now())
                    last_completion=tick
                    # Publish throughput metadata and shared progress together.
                    with progress.lock:
                        budget.atomic_json(budget.root/'records'/(entry['name']+'.json'),record)
                        progress.completed[entry['name']]=record
                        progress.active.pop(entry['name'],None)
                        progress('tile_completed',1,1,force=True)
                    submit_next()
        finally:
            for future in pending:
                future.cancel()


def cleanup_temporary(budget):
    """Only remove known job-owned partial names, after all worker threads exit."""
    for path in (budget.root/'temporary').iterdir():
        if path.name in ('source-download.tif.part','compressed-output.tif.part') or re.fullmatch(
            r'm_\d{7}_(ne|nw|se|sw)_\d{2}_060_\d{8}\.tif\.(source|compressed)\.part',path.name):
            budget.remove(path)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_ROOT,
                        help='Folder containing all job data (default: repository data/nys-naip-2022-rgb-060m)')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--check-running',action='store_true')
    parser.add_argument('--workers',type=int,choices=range(1,5),default=DEFAULT_WORKERS)
    return parser.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    root=args.output_dir.resolve()
    if args.check_running:
        if not root.exists():
            return 1
        try:
            with job_lock(root):
                return 1
        except RuntimeError:
            return 0
    root.mkdir(parents=True,exist_ok=True)
    for name in ('temporary','imagery','records','metadata','runtime'):
        (root/name).mkdir(exist_ok=True)
    for zone in (17,18,19):
        (root/'imagery'/f'utm{zone}').mkdir(exist_ok=True)
    for name in ('TMP','TEMP','TMPDIR','GDAL_TMPDIR','CPL_TMPDIR'):
        os.environ[name]=str(root/'temporary')
    with job_lock(root):
        budget=Budget(root)
        if args.prepare or not (root/'manifest.json').exists():
            manifest=prepare(budget)
            print(json.dumps({k:v for k,v in manifest.items() if k!='tiles'},indent=2),flush=True)
            if args.prepare:
                return 0
        else:
            manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
            if (manifest.get('dataset')!='NY_NAIP_2022_9986'
                or manifest.get('resolution_m_per_pixel')!=0.6
                or manifest.get('bands')!=[1,2,3]
                or manifest.get('jpeg_quality_candidates')!=list(QUALITIES)
                or manifest.get('folder_cap_bytes')!=CAP
                or manifest.get('tile_count')!=len(manifest.get('tiles',[]))
                or manifest.get('tile_count')!=3805):
                raise ValueError('Saved manifest is incompatible with this worker')
            for name, expected_hash in manifest['metadata_sha256'].items():
                path=budget.contained(root/'metadata'/name)
                if not path.exists() or sha256(path)!=expected_hash:
                    raise ValueError('Provider metadata snapshot has changed')
        completed={}
        current=None
        progress=Progress(budget,manifest,completed)
        progress.workers=args.workers
        try:
            check_pause(root)
            progress('verifying_existing_outputs',0,manifest['tile_count'],force=True,state='verifying')
            for scanned,entry in enumerate(manifest['tiles'],1):
                check_pause(root)
                record=verified_record(entry,budget)
                if record:
                    completed[entry['name']]=record
                progress('verifying_existing_outputs',scanned,manifest['tile_count'],state='verifying')
            tiles=manifest['tiles']
            if args.pilot:
                chosen=['m_4207318_se_18_060_20220731.tif','m_4007309_ne_18_060_20220719.tif']
                northern=next(e['name'] for e in tiles if e['name'].startswith('m_44074'))
                chosen.append(northern)
                tiles=[next(e for e in tiles if e['name']==name) for name in chosen]
            write_status(budget,manifest,'running',completed)
            cleanup_temporary(budget)
            pending_tiles=[entry for entry in tiles if entry['name'] not in completed]
            if args.limit:
                pending_tiles=pending_tiles[:args.limit]
            run_tiles(pending_tiles,budget,progress,args.workers)
            state='complete' if len(completed)==manifest['tile_count'] else 'pilot_complete' if args.pilot else 'batch_complete'
            write_status(budget,manifest,state,completed)
            if state=='complete':
                budget.atomic_json(root/'completion.json',{'completed_utc':now(),'tile_count':len(completed),
                    'output_bytes':sum(r['output_bytes'] for r in completed.values()),'manifest_sha256':sha256(root/'manifest.json')})
            elif args.pilot:
                budget.atomic_json(root/'pilot.json',{'completed_utc':now(),'tiles':[completed[e['name']] for e in tiles],
                    'projected_GB_at_pilot_mean_ratio':manifest['raw_rgb_bytes']/
                    (sum(e['raw_rgb_bytes'] for e in tiles)/sum(completed[e['name']]['output_bytes'] for e in tiles))/1e9})
            return 0
        except BaseException as exc:
            cleanup_temporary(budget)
            state='paused' if isinstance(exc,(PauseRequested,KeyboardInterrupt)) else 'failed'
            write_status(budget,manifest,state,completed,progress.current,str(exc))
            event(budget,state,tile=progress.current,error=str(exc))
            return 2


if __name__=='__main__':
    sys.exit(main())
