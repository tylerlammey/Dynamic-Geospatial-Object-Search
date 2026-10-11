"""Lazy native-pixel chip planning and common WGS84 footprints for NYS TIFFs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
from shapely import to_wkb
from shapely.geometry import Polygon

from .download_nys_naip import sha256
from .paths import contained


@dataclass(frozen=True)
class Grid:
    size: int = 256
    stride: int = 192
    min_valid_fraction: float = 0.95

    def __post_init__(self):
        if not isinstance(self.size, int) or not isinstance(self.stride, int) or not 0 < self.stride <= self.size:
            raise ValueError('Require integer pixels with 0 < stride <= size')
        if not 0 <= self.min_valid_fraction <= 1:
            raise ValueError('Minimum valid fraction must be between zero and one')

    def starts(self, length: int) -> Iterator[int]:
        if length < self.size:
            return
        final = length - self.size
        yield from range(0, final + 1, self.stride)
        if final % self.stride:
            yield final

    def axis_count(self, length: int) -> int:
        if length < self.size:
            return 0
        final = length - self.size
        return final // self.stride + 1 + int(final % self.stride != 0)

    def count(self, width: int, height: int) -> int:
        return self.axis_count(width) * self.axis_count(height)

    def windows(self, width: int, height: int) -> Iterator[Window]:
        for y in self.starts(height):
            for x in self.starts(width):
                yield Window(x, y, self.size, self.size)

    @property
    def scale_id(self) -> str:
        return f'native060cm-s{self.size}-st{self.stride}'


@dataclass(frozen=True)
class Source:
    path: Path
    uri: str
    width: int
    height: int
    crs: str
    transform: tuple
    sha256: str
    size_bytes: int
    jpeg_quality: int

    @property
    def shard_id(self):
        return self.path.stem


def load_inventory(job_root: Path) -> tuple[list[Source], str]:
    """Read the downloader's completed compatibility set, not arbitrary TIFFs."""
    job_root = job_root.resolve()
    manifest_path = job_root / 'manifest.json'
    manifest_hash = sha256(manifest_path)
    completion = json.loads((job_root / 'completion.json').read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (completion.get('manifest_sha256') != manifest_hash or manifest.get('tile_count') != 3805
        or completion.get('tile_count') != 3805 or len(manifest.get('tiles', [])) != 3805
        or manifest.get('resolution_m_per_pixel') != .6 or manifest.get('bands') != [1,2,3]):
        raise ValueError('NYS acquisition is incomplete or its manifest is incompatible')
    sources = []
    for entry in manifest['tiles']:
        record = json.loads((job_root / 'records' / (entry['name']+'.json')).read_text(encoding='utf-8'))
        expected = f"imagery/utm{entry['zone']}/{entry['name']}"
        if record.get('output') != expected or record.get('source_url') != entry['url']:
            raise ValueError(f"Incompatible provenance for {entry['name']}")
        path = contained(job_root / 'imagery', job_root / expected)
        if (not path.is_file() or path.stat().st_size != record['output_bytes']
            or record['width'] != entry['width'] or record['height'] != entry['height']
            or record['crs'] != entry['crs'] or not np.allclose(record['transform'],entry['transform'],rtol=0,atol=1e-6)):
            raise ValueError(f'Imagery differs from downloader records: {path}')
        sources.append(Source(path,path.relative_to(job_root/'imagery').as_posix(),
                              entry['width'],entry['height'],entry['crs'],tuple(entry['transform']),
                              record['output_sha256'],record['output_bytes'],record['jpeg_quality']))
    if len({s.uri for s in sources}) != len(sources) or len({s.shard_id for s in sources}) != len(sources):
        raise ValueError('Duplicate source identity')
    return sorted(sources,key=lambda source:source.uri), manifest_hash


def validate_source(dataset, source: Source):
    if (dataset.count != 3 or dataset.dtypes != ('uint8',)*3 or dataset.crs is None
        or dataset.crs.to_string() != source.crs or dataset.shape != (source.height,source.width)
        or not np.allclose(tuple(dataset.transform)[:6],source.transform,rtol=0,atol=1e-6)
        or not np.allclose(dataset.res,(.6,.6),rtol=0,atol=1e-9)):
        raise ValueError(f'Source pixel grid or RGB contract changed: {source.uri}')


def read_rgb(dataset, window: Window) -> Image.Image:
    if (window.col_off < 0 or window.row_off < 0 or window.width <= 0 or window.height <= 0
        or window.col_off + window.width > dataset.width or window.row_off + window.height > dataset.height):
        raise ValueError('Chip window escapes the source raster')
    pixels = dataset.read((1,2,3),window=window,boundless=False)
    if pixels.dtype != np.uint8 or pixels.shape != (3,int(window.height),int(window.width)):
        raise ValueError('Expected native uint8 RGB pixels')
    return Image.fromarray(np.moveaxis(pixels,0,-1))


class Footprints:
    """Reuse a transformer and vectorized densified boundary for each raster."""
    def __init__(self, dataset, size: int):
        self.transform = dataset.transform
        self.crs_wkt = dataset.crs.to_wkt()
        self.transformer = Transformer.from_crs(dataset.crs,'EPSG:4326',always_xy=True)
        edge = np.linspace(0,size,9,endpoint=False)
        self.cols = np.concatenate((edge,np.full(9,size),size-edge,np.zeros(9)))
        self.rows = np.concatenate((np.zeros(9),edge,np.full(9,size),size-edge))
        self.size = size

    def metadata(self, window: Window) -> dict:
        cols,rows = self.cols+window.col_off,self.rows+window.row_off
        t = self.transform
        native_x,native_y = t.a*cols+t.b*rows+t.c,t.d*cols+t.e*rows+t.f
        lon,lat = self.transformer.transform(native_x,native_y)
        center_x,center_y = window.col_off+self.size/2,window.row_off+self.size/2
        center = self.transformer.transform(t.a*center_x+t.b*center_y+t.c,
                                            t.d*center_x+t.e*center_y+t.f)
        corners = [(window.col_off,window.row_off),(window.col_off+self.size,window.row_off),
                   (window.col_off+self.size,window.row_off+self.size),(window.col_off,window.row_off+self.size)]
        native_corners = [(t.a*x+t.b*y+t.c,t.d*x+t.e*y+t.f) for x,y in corners]
        return {'native_minx':min(p[0] for p in native_corners),'native_miny':min(p[1] for p in native_corners),
                'native_maxx':max(p[0] for p in native_corners),'native_maxy':max(p[1] for p in native_corners),
                'west':float(min(lon)),'south':float(min(lat)),'east':float(max(lon)),'north':float(max(lat)),
                'centroid_lon':center[0],'centroid_lat':center[1],'crs_wkt':self.crs_wkt,
                'geometry':to_wkb(Polygon(zip(lon,lat))),
                'window_transform':list(rasterio.windows.transform(window,t))[:6]}


def summarize(sources: list[Source], grid: Grid) -> dict:
    count = sum(grid.count(source.width,source.height) for source in sources)
    return {'source_tiles':len(sources),'candidate_chips':count,'chip_size':grid.size,'stride':grid.stride,
            'nominal_chip_width_m':grid.size*.6,'nominal_overlap_percent':100*(1-grid.stride/grid.size),
            'min_valid_fraction':grid.min_valid_fraction,'geometry_crs':'EPSG:4326',
            'vector_bytes_before_mask_filtering_1024d':count*1024*2,
            'note':'Includes source-buffer/final-edge overlap; mask-filtered chip count is measured while embedding.'}
