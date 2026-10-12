"""Frontend-only photographic display pyramid; original TIFFs remain read-only."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor,CancelledError
import math
import threading
import time

import numpy as np
import rasterio
from rasterio.enums import ColorInterp,Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine,from_origin
from rasterio.warp import reproject,transform_bounds

from imagery import PHOTO_SIZE
from PIL import Image


def composite(region,destination):
    # A partially transparent provider edge must not erase valid imagery from
    # an earlier overlapping provider source.
    background=Image.fromarray(region.transpose(1,2,0))
    foreground=Image.fromarray(destination.transpose(1,2,0))
    region[:]=np.asarray(Image.alpha_composite(background,foreground)).transpose(2,0,1)


def prepare_overview(renderer,progress,workers=4,cancel=None,should_stop=None):
    cancel=cancel or threading.Event();start=time.monotonic()
    inventory=renderer.inventory
    west,south,east,north=transform_bounds(4326,3857,*inventory.bounds,densify_pts=21)
    resolution=128.
    width=max(1,math.ceil((east-west)/resolution));height=max(1,math.ceil((north-south)/resolution))
    if width*height>50_000_000:raise ValueError('Statewide display overview exceeds the bounded pixel limit')
    transform=from_origin(west,north,resolution,resolution)
    # At NYS extent this single display mosaic is about 150 MB in RAM. It is
    # downsampled display data, never a native-resolution statewide image.
    mosaic=np.zeros((4,height,width),dtype='uint8')
    def read(source):return np.asarray(renderer.photograph(source)).transpose(2,0,1)
    sources=inventory.sources
    progress({'state':'preparing','completed_sources':0,'total_sources':len(sources),'elapsed_seconds':0,'width':width,'height':height})
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='overview') as pool:
        pending=deque();next_index=0
        # Keep at most workers photographic arrays/futures alive, in a stable
        # source order, rather than queueing all 3,805 images into RAM.
        for _ in range(min(workers,len(sources))):
            pending.append((sources[next_index],pool.submit(read,sources[next_index])));next_index+=1
        completed=0
        while pending:
            if cancel.is_set() or renderer.closed or (should_stop and should_stop()):
                cancel.set()
                for _,future in pending:future.cancel()
                raise CancelledError('Overview preparation stopped; cached source overviews can be resumed')
            source,future=pending.popleft();pixels=future.result()
            bounds=transform_bounds(4326,3857,*source.bounds,densify_pts=8)
            left=max(0,int(math.floor((bounds[0]-west)/resolution))-1);right=min(width,int(math.ceil((bounds[2]-west)/resolution))+1)
            top=max(0,int(math.floor((north-bounds[3])/resolution))-1);bottom=min(height,int(math.ceil((north-bounds[1])/resolution))+1)
            if right>left and bottom>top:
                destination=np.zeros((4,bottom-top,right-left),dtype='uint8')
                src_transform=Affine(*source.transform)*Affine.scale(source.width/PHOTO_SIZE,source.height/PHOTO_SIZE)
                with rasterio.Env(GDAL_PAM_ENABLED=False,GDAL_CACHEMAX=32_000_000,GDAL_NUM_THREADS='1'):
                    reproject(pixels,destination,src_transform=src_transform,src_crs=source.crs,dst_transform=transform*Affine.translation(left,top),dst_crs='EPSG:3857',src_alpha=4,dst_alpha=4,resampling=Resampling.average,num_threads=1)
                region=mosaic[:,top:bottom,left:right]
                # Stable provider precedence while preserving transparent edges.
                composite(region,destination)
            completed+=1
            elapsed=time.monotonic()-start
            progress({'state':'preparing','completed_sources':completed,'total_sources':len(sources),'elapsed_seconds':round(elapsed,1),
                      'eta_seconds':round(elapsed/completed*(len(sources)-completed),1),'current_source':source.uri})
            if next_index<len(sources):
                pending.append((sources[next_index],pool.submit(read,sources[next_index])));next_index+=1
    if cancel.is_set():raise CancelledError('Overview preparation cancelled')
    progress({'state':'publishing','completed_sources':len(sources),'total_sources':len(sources),'elapsed_seconds':round(time.monotonic()-start,1)})
    generation=renderer.cache.generation
    with rasterio.Env(GDAL_PAM_ENABLED=False,GDAL_CACHEMAX=64_000_000,GDAL_NUM_THREADS='1'):
        with MemoryFile() as memory:
            with memory.open(driver='GTiff',width=width,height=height,count=4,dtype='uint8',crs='EPSG:3857',transform=transform,
                             tiled=True,blockxsize=256,blockysize=256,compress='deflate',predictor=2) as ds:
                ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
                ds.write(mosaic)
                factors=[f for f in (2,4,8,16,32) if min(width,height)//f>=1]
                if factors:ds.build_overviews(factors,Resampling.average)
            renderer.cache.write(renderer.overview_key,memory.read(),generation)
    renderer.display_epoch+=1
    progress({'state':'complete','completed_sources':len(sources),'total_sources':len(sources),'elapsed_seconds':round(time.monotonic()-start,1),
              'width':width,'height':height,'resolution_m':resolution,'ready':True})
