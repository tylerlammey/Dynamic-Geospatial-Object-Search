"""Additive NYS NAIP gap-fill inventory; immutable base data stays separate."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
from .download_nys_naip import sha256
from .paths import JOB,contained
from .tiling import Source

BASE_DATASET='nys_naip_2022_rgb060'
DATASET='nys_naip_2021_release_gapfill_rgb060'
NAME='naip-2021-release-gapfill'
REMOTE='https://coastalimagery.blob.core.windows.net/digitalcoast/NY_NAIP_2021_9664/'
RELATIVE=Path('supplements')/NAME


def directory(job=JOB):return contained(Path(job),Path(job)/RELATIVE)


class ScopedBudget:
    """A supplement path scope backed by the primary job's one byte budget."""
    def __init__(self,budget,root):
        self.budget=budget;self.root=budget.contained(root);self.lock=budget.lock
    def contained(self,path):return self.budget.contained(contained(self.root,path))
    def check(self,path,size):return self.budget.check(self.contained(path),size)
    def account(self,path,size):return self.budget.account(self.contained(path),size)
    def write_bytes(self,path,content,append=False):return self.budget.write_bytes(self.contained(path),content,append)
    def atomic_json(self,path,value):return self.budget.atomic_json(self.contained(path),value)
    def replace(self,source,destination):return self.budget.replace(self.contained(source),self.contained(destination))
    def remove(self,path):return self.budget.remove(self.contained(path))
    def __getattr__(self,name):return getattr(self.budget,name)


def load_inventory(job=JOB,require_complete=True):
    root=directory(job);manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));digest=sha256(manifest_path)
    if (manifest.get('dataset_id')!=DATASET or manifest.get('release_year')!=2021
        or manifest.get('bands')!=[1,2,3] or manifest.get('resolution_m_per_pixel')!=.6
        or len(manifest.get('tiles',[]))!=manifest.get('tile_count')):
        raise ValueError('Incompatible gap-fill manifest')
    if require_complete:
        completed=json.loads((root/'completion.json').read_text(encoding='utf-8'))
        if completed.get('manifest_sha256')!=digest or completed.get('tile_count')!=manifest['tile_count']:
            raise ValueError('Gap-fill acquisition is incomplete')
    sources=[]
    for tile in manifest['tiles']:
        name=tile['name']
        if not re.fullmatch(r'm_\d{7}_[ns][ew]_18_060_\d{8}\.tif',name) or tile['url']!=REMOTE+name:
            raise ValueError('Gap-fill source must be an allowlisted NOAA tile')
        record_path=root/'records'/f'{name}.json'
        if not record_path.exists():
            if require_complete:raise ValueError('Missing gap-fill source record')
            continue
        record=json.loads(record_path.read_text(encoding='utf-8'))
        expected=f'imagery/utm18/{name}';path=contained(root/'imagery',root/expected)
        if (record.get('output')!=expected or record.get('source_url')!=tile['url']
            or record.get('dataset_id')!=DATASET or record.get('release_year')!=2021
            or not path.is_file() or path.stat().st_size!=record['output_bytes']
            or (record['width'],record['height'])!=(tile['width'],tile['height'])
            or record['crs']!='EPSG:26918'
            or not np.allclose(record['transform'],tile['transform'],rtol=0,atol=1e-6)):
            raise ValueError(f'Invalid gap-fill provenance: {name}')
        sources.append(Source(path,path.relative_to(root/'imagery').as_posix(),tile['width'],tile['height'],
            tile['crs'],tuple(tile['transform']),record['output_sha256'],record['output_bytes'],record['jpeg_quality']))
    if len({s.uri for s in sources})!=len(sources):raise ValueError('Duplicate gap-fill source')
    return sorted(sources,key=lambda source:source.uri),digest


@dataclass(frozen=True)
class Collection:
    dataset_id:str
    directory:Path
    imagery:Path
    release_year:int
    uri_prefix:str=''

    def qualify(self,uri):return self.uri_prefix+uri
    def acquisition_date(self,uri):
        match=re.search(r'_(\d{4})(\d{2})(\d{2})\.tif$',uri)
        return '-'.join(match.groups()) if match else None


def collections(job=JOB,include_supplements=True):
    job=Path(job).resolve()
    result=[Collection(BASE_DATASET,job/'embeddings/georsclip-vith14-s256-st192',job/'imagery',2022)]
    root=directory(job);collection=root/'embeddings/georsclip-vith14-s256-st192'
    if include_supplements and (root/'completion.json').exists() and (collection/'collection.json').exists():
        load_inventory(job)
        result.append(Collection(DATASET,collection,root/'imagery',2021,f'{RELATIVE.as_posix()}/'))
    return result
