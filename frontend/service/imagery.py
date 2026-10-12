"""Read-only native NAIP sources -> frontend-only Web Mercator display tiles."""
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from concurrent.futures import CancelledError
from shapely.geometry import box
from shapely.strtree import STRtree

from storage import contained

EARTH = 20037508.342789244
SIZE = 256
DISPLAY_VERSION = 'photographic-v2'
PHOTO_SIZE = 512


def tile_bounds(z,x,y,geographic=False):
    if any(type(n) is not int for n in (z,x,y)) or not 0<=z<=20 or not 0<=x<2**z or not 0<=y<2**z:
        raise ValueError('Invalid XYZ tile')
    span=2*EARTH/(2**z)
    bounds=(-EARTH+x*span,EARTH-(y+1)*span,-EARTH+(x+1)*span,EARTH-y*span)
    return transform_bounds(3857,4326,*bounds,densify_pts=2) if geographic else bounds


def lonlat_tile(lon,lat,z):
    if not math.isfinite(lon) or not math.isfinite(lat) or not -180<=lon<=180 or not -85<=lat<=85:
        raise ValueError('Invalid longitude/latitude')
    n=2**z
    return z,min(n-1,int((lon+180)/360*n)),min(n-1,int((1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n))


@dataclass(frozen=True)
class Source:
    uri:str
    path:Path
    width:int
    height:int
    crs:str
    transform:tuple
    bounds:tuple
    sha256:str
    bytes:int
    mtime:int


class Inventory:
    def __init__(self,job,strict=True):
        self.job=Path(job).resolve()
        manifest_path=self.job/'manifest.json'
        blob=manifest_path.read_bytes();manifest=json.loads(blob)
        if strict:
            if manifest.get('tile_count')!=3805 or len(manifest.get('tiles',[]))!=3805 or manifest.get('bands')!=[1,2,3] or manifest.get('resolution_m_per_pixel')!=.6:
                raise ValueError('Expected completed NYS 2022 native RGB inventory')
            completion=json.loads((self.job/'completion.json').read_text(encoding='utf-8'))
            if completion.get('manifest_sha256')!=hashlib.sha256(blob).hexdigest():raise ValueError('Source manifest differs from completed acquisition')
        self.fingerprint=hashlib.sha256(blob).hexdigest()[:20]
        sources=[]
        for entry in manifest['tiles']:
            name=entry['name']
            if Path(name).name!=name:raise ValueError('Invalid source name')
            record=json.loads((self.job/'records'/f'{name}.json').read_text(encoding='utf-8'))
            path=contained(self.job/'imagery',self.job/record['output'])
            stat=path.stat()
            if stat.st_size!=record['output_bytes']:raise ValueError('Source size differs from provenance')
            if strict:
                expected=f"imagery/utm{entry['zone']}/{name}"
                if record['output']!=expected or record['source_url']!=entry['url'] or record['crs']!=entry['crs'] or (record['width'],record['height'])!=(entry['width'],entry['height']) or not np.allclose(record['transform'],entry['transform'],rtol=0,atol=1e-6):
                    raise ValueError('Source record differs from manifest')
            sources.append(Source(path.relative_to(self.job/'imagery').as_posix(),path,record['width'],record['height'],record['crs'],tuple(record['transform']),tuple(record['bounds_wgs84']),record['output_sha256'],stat.st_size,stat.st_mtime_ns))
        self.sources=sorted(sources,key=lambda s:s.uri)
        self.by_uri={s.uri:s for s in self.sources}
        if len(self.by_uri)!=len(self.sources):raise ValueError('Duplicate sources')
        self.tree=STRtree([box(*s.bounds) for s in self.sources])
        self.bounds=(min(s.bounds[0] for s in sources),min(s.bounds[1] for s in sources),max(s.bounds[2] for s in sources),max(s.bounds[3] for s in sources))

    def intersect(self,bounds):
        return [self.sources[int(i)] for i in sorted(self.tree.query(box(*bounds)))]

    def check(self,source):
        stat=source.path.stat()
        if stat.st_size!=source.bytes or stat.st_mtime_ns!=source.mtime:raise ValueError('Original imagery changed; restart to verify inventory')

    def public(self):
        return {'bounds':self.bounds,'count':len(self.sources),'fingerprint':self.fingerprint,'source_resolution':.6,'year':2022}


def png(image):
    stream=io.BytesIO();image.save(stream,format='PNG',compress_level=3);return stream.getvalue()


class TileRenderer:
    def __init__(self,inventory,cache,workers=2):
        self.inventory=inventory;self.cache=cache
        self.slots=threading.BoundedSemaphore(workers)
        self.closed=False
        self.metrics={'tiles_rendered':0,'source_opens':0,'last_tile_seconds':0}
        self.metric_lock=threading.Lock()
        self.photo_locks=[threading.Lock() for _ in range(64)]
        self.display_epoch=0
        self.overview_key=f'{inventory.fingerprint}/{DISPLAY_VERSION}/state.tif'

    def overview_ready(self):return self.cache.exists(self.overview_key)

    def photograph(self,source):
        """One reusable, mask-aware averaged photographic source overview."""
        key=f'{self.inventory.fingerprint}/{DISPLAY_VERSION}/sources/{Path(source.uri).stem}.png'
        cached=self.cache.read(key)
        if cached:return Image.open(io.BytesIO(cached)).convert('RGBA')
        with self.photo_locks[hash(source.uri)%len(self.photo_locks)]:
            cached=self.cache.read(key)
            if cached:return Image.open(io.BytesIO(cached)).convert('RGBA')
            generation=self.cache.generation
            self.inventory.check(source)
            with rasterio.Env(GDAL_PAM_ENABLED=False,GDAL_CACHEMAX=64_000_000,GDAL_NUM_THREADS='1'):
                with rasterio.open(source.path,sharing=False) as ds:
                    # Native RasterIO averaging respects the internal validity
                    # mask and avoids a redundant same-CRS warp pass.
                    rgb=ds.read((1,2,3),out_shape=(3,PHOTO_SIZE,PHOTO_SIZE),resampling=Resampling.average)
                    alpha=ds.dataset_mask(out_shape=(PHOTO_SIZE,PHOTO_SIZE),resampling=Resampling.average)
                    pixels=np.concatenate((rgb,alpha[None]),axis=0)
            image=Image.fromarray(pixels.transpose(1,2,0))
            self.cache.write(key,png(image),generation)
            return image

    def tile(self,z,x,y,cancel=None):
        import time
        if cancel is not None and cancel.is_set():raise CancelledError('Obsolete viewport tile')
        bounds=tile_bounds(z,x,y)
        ready=self.overview_ready()
        key=f'{self.inventory.fingerprint}/{DISPLAY_VERSION}/tiles-{self.display_epoch}/{z}/{x}/{y}.png'
        cached=self.cache.read(key)
        if cached:return cached
        with self.slots:
            if self.closed:raise RuntimeError('Imagery service is closing')
            cached=self.cache.read(key)
            if cached:return cached
            started=time.perf_counter();generation=self.cache.generation
            target=from_bounds(*bounds,SIZE,SIZE)
            result=Image.new('RGBA',(SIZE,SIZE))
            if z<=10:
                if not ready:
                    # Preparation is required; statewide requests must not
                    # scan thousands of original TIFFs.
                    return png(result)
                with self.cache.file(self.overview_key) as overview:
                    if overview is None:return png(result)
                    with rasterio.Env(GDAL_PAM_ENABLED=False,GDAL_CACHEMAX=32_000_000,GDAL_NUM_THREADS='1'):
                        with rasterio.open(overview,sharing=False) as ds:
                            with WarpedVRT(ds,crs='EPSG:3857',transform=target,width=SIZE,height=SIZE,resampling=Resampling.bilinear,warp_mem_limit=16) as vrt:
                                result=Image.fromarray(vrt.read((1,2,3,4)).transpose(1,2,0))
                blob=png(result);self.cache.write(key,blob,generation)
                with self.metric_lock:
                    self.metrics['tiles_rendered']+=1;self.metrics['last_tile_seconds']=round(time.perf_counter()-started,3)
                return blob
            sources=self.inventory.intersect(tile_bounds(z,x,y,geographic=True))
            for source in sources:
                if cancel is not None and cancel.is_set():raise CancelledError('Obsolete viewport tile')
                self.inventory.check(source)
                with rasterio.Env(GDAL_PAM_ENABLED=False,GDAL_CACHEMAX=32_000_000,GDAL_NUM_THREADS='1'):
                    if z<14:
                        image=np.asarray(self.photograph(source)).transpose(2,0,1)
                        destination=np.zeros((4,SIZE,SIZE),dtype='uint8')
                        src_transform=Affine(*source.transform)*Affine.scale(source.width/PHOTO_SIZE,source.height/PHOTO_SIZE)
                        reproject(image,destination,src_transform=src_transform,src_crs=source.crs,dst_transform=target,dst_crs='EPSG:3857',src_alpha=4,dst_alpha=4,resampling=Resampling.bilinear,num_threads=1)
                    else:
                        with rasterio.open(source.path,sharing=False) as ds:
                            with WarpedVRT(ds,crs='EPSG:3857',transform=target,width=SIZE,height=SIZE,add_alpha=True,resampling=Resampling.bilinear,warp_mem_limit=32) as vrt:
                                destination=vrt.read((1,2,3,4))
                    result.alpha_composite(Image.fromarray(destination.transpose(1,2,0)))
            blob=png(result);self.cache.write(key,blob,generation)
            with self.metric_lock:
                self.metrics['tiles_rendered']+=1;self.metrics['source_opens']+=len(sources);self.metrics['last_tile_seconds']=round(time.perf_counter()-started,3)
            return blob

    def inspect(self,lon,lat):
        lonlat_tile(lon,lat,18)
        sources=self.inventory.intersect((lon-.0000001,lat-.0000001,lon+.0000001,lat+.0000001))
        for source in reversed(sources):
            self.inventory.check(source)
            with rasterio.Env(GDAL_PAM_ENABLED=False):
                with rasterio.open(source.path,sharing=False) as ds:
                    nx,ny=Transformer.from_crs(4326,ds.crs,always_xy=True).transform(lon,lat)
                    row,col=ds.index(nx,ny)
                    if not 0<=col<ds.width or not 0<=row<ds.height:continue
                    if ds.dataset_mask(window=Window(col,row,1,1))[0,0]==0:continue
                    x=max(0,min(ds.width-256,col-128));y=max(0,min(ds.height-256,row-128))
                    from geospatial_search.tiling import Footprints
                    meta=Footprints(ds,256).metadata(Window(x,y,256,256))
                    from shapely import from_wkb
                    geometry=from_wkb(meta.pop('geometry')).__geo_interface__
                    return {'rank':None,'score':None,'shard_id':Path(source.uri).stem,'embedding_id':None,'chip_id':f'map:{source.uri}:{x}:{y}',
                            'source_uri':source.uri,'source_sha256':source.sha256,'pixel_x':x,'pixel_y':y,'pixel_width':256,'pixel_height':256,'geometry':geometry,**meta}
        raise ValueError('No valid downloaded imagery at this location')

    def preview(self,row,context=False):
        source=self.inventory.by_uri.get(row['source_uri'])
        if not source or row.get('source_sha256')!=source.sha256:raise ValueError('Result source differs from provenance')
        x,y,w,h=(row[k] for k in ('pixel_x','pixel_y','pixel_width','pixel_height'))
        if any(type(n) is not int for n in (x,y,w,h)) or w!=256 or h!=256 or x<0 or y<0 or x+w>source.width or y+h>source.height:raise ValueError('Invalid native chip window')
        key=f'{self.inventory.fingerprint}/previews/{Path(source.uri).stem}-{x}-{y}-{int(context)}.png'
        cached=self.cache.read(key)
        if cached:return cached
        generation=self.cache.generation
        self.inventory.check(source)
        with self.slots, rasterio.Env(GDAL_PAM_ENABLED=False,GDAL_CACHEMAX=32_000_000,GDAL_NUM_THREADS='1'):
            with rasterio.open(source.path,sharing=False) as ds:
                cx=max(0,x-256) if context else x;cy=max(0,y-256) if context else y
                cw=min(source.width-cx,768) if context else w;ch=min(source.height-cy,768) if context else h
                window=Window(cx,cy,cw,ch)
                pixels=ds.read((1,2,3),window=window)
                rgba=np.concatenate((pixels,ds.dataset_mask(window=window)[None,:,:]),axis=0).transpose(1,2,0)
                image=Image.fromarray(rgba)
                if context:
                    ImageDraw.Draw(image).rectangle((x-cx,y-cy,x-cx+w,y-cy+h),outline='#ffcf62',width=5)
                    image.thumbnail((384,384),Image.Resampling.LANCZOS)
                blob=png(image)
        self.cache.write(key,blob,generation)
        return blob

    def close(self):self.closed=True
