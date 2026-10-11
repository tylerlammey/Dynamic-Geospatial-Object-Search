"""Atomic vector/GeoParquet/run bundles and bounded collection search."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS

from .download_nys_naip import sha256
from .paths import contained

BUNDLE_FILES = ('embeddings.f16.npy','chips.geoparquet','run.json')
METADATA_SCHEMA = pa.schema([
    ('embedding_id',pa.uint64()),('dataset_id',pa.string()),('chip_id',pa.string()),
    ('source_uri',pa.string()),('source_tile_id',pa.string()),('source_sha256',pa.string()),
    ('pixel_x',pa.int32()),('pixel_y',pa.int32()),('pixel_width',pa.int32()),('pixel_height',pa.int32()),
    *[(name,pa.float64()) for name in ('native_minx','native_miny','native_maxx','native_maxy',
                                     'west','south','east','north','centroid_lon','centroid_lat','valid_fraction')],
    ('crs_wkt',pa.string()),('scale_id',pa.string()),('rgb_band_order',pa.string()),
    ('model_id',pa.string()),('model_fingerprint',pa.string()),('preprocess_fingerprint',pa.string()),
    ('run_fingerprint',pa.string()),('window_transform',pa.list_(pa.float64())),('geometry',pa.binary()),
])


def metadata_table(rows: list[dict]) -> pa.Table:
    table = pa.Table.from_pylist(rows,schema=METADATA_SCHEMA)
    ids = table['embedding_id'].to_numpy()
    if not np.array_equal(ids,np.arange(len(rows))):
        raise ValueError('Metadata embedding IDs must equal vector row numbers')
    table = table.set_column(table.schema.get_field_index('embedding_id'),
                             pa.field('embedding_id',pa.uint64()),pa.array(ids,type=pa.uint64()))
    geometry = {'encoding':'WKB','geometry_types':['Polygon'],'crs':CRS.from_epsg(4326).to_json_dict()}
    if rows:
        geometry['bbox'] = [min(r['west'] for r in rows),min(r['south'] for r in rows),
                            max(r['east'] for r in rows),max(r['north'] for r in rows)]
    geo = {'version':'1.0.0','primary_column':'geometry','columns':{'geometry':geometry}}
    metadata = dict(table.schema.metadata or {})
    metadata[b'geo'] = json.dumps(geo,separators=(',',':')).encode()
    return table.replace_schema_metadata(metadata)


def publish_bundle(budget, directory: Path, features: np.ndarray, rows: list[dict], run: dict):
    """All bytes are encoded in RAM, budget-checked, then the directory is published."""
    directory = budget.contained(directory)
    if directory.exists():
        raise FileExistsError(f'Completed bundle already exists: {directory}')
    if features.shape != (len(rows),run['embedding_dim']) or not np.isfinite(features).all():
        raise ValueError('Vector/metadata shape or numeric validity mismatch')
    if not np.allclose(np.linalg.norm(features.astype(np.float32),axis=1),1,atol=.002):
        raise ValueError('Embeddings must be L2-normalized before storage')
    table = metadata_table(rows)
    arrays = io.BytesIO()
    np.save(arrays,np.asarray(features,dtype=np.float16),allow_pickle=False)
    parquet = pa.BufferOutputStream()
    pq.write_table(table,parquet,compression='zstd',row_group_size=4096)
    blobs = {'embeddings.f16.npy':arrays.getvalue(),'chips.geoparquet':parquet.getvalue().to_pybytes()}
    stage = directory.parent / ('.'+directory.name+'.staging')
    if stage.exists():
        raise FileExistsError(f'Unfinished staging bundle exists: {stage}')
    stage.mkdir(parents=True)
    for name,blob in blobs.items():
        budget.write_bytes(stage/name,blob)
    run = dict(run,chip_count=len(rows),embedding_dtype='float16',
               vectors_sha256=sha256(stage/'embeddings.f16.npy'),metadata_sha256=sha256(stage/'chips.geoparquet'))
    budget.atomic_json(stage/'run.json',run)
    validate_bundle(stage,run['run_fingerprint'],verify_hashes=True)
    # Directory publication is atomic; data files retain their contents and sizes.
    old_paths = [(stage/name,(stage/name).stat().st_size) for name in BUNDLE_FILES]
    stage.replace(directory)
    with budget.lock:
        for old_path,size in old_paths:
            budget.account(old_path,0)
            budget.sizes.pop(old_path.resolve(),None)
            budget.account(directory/old_path.name,size)
    return run


def discard_staging(budget, shards: Path):
    if not shards.exists():
        return
    for directory in shards.iterdir():
        if directory.name.startswith('.') and directory.name.endswith('.staging') and directory.is_dir():
            if any(p.name not in set(BUNDLE_FILES)|{'run.json.new'} or not p.is_file() for p in directory.iterdir()):
                raise ValueError(f'Unexpected files in staging directory: {directory}')
            for path in list(directory.iterdir()):
                budget.remove(path)
            directory.rmdir()


def validate_bundle(directory: Path, run_fingerprint: str, *, verify_hashes=False) -> dict:
    run = json.loads((directory/'run.json').read_text(encoding='utf-8'))
    if run.get('run_fingerprint') != run_fingerprint or run.get('embedding_dtype') != 'float16':
        raise ValueError(f'Incompatible embedding bundle: {directory}')
    vectors = np.load(directory/'embeddings.f16.npy',mmap_mode='r',allow_pickle=False)
    if vectors.dtype != np.float16 or vectors.shape != (run['chip_count'],run['embedding_dim']):
        raise ValueError(f'Invalid vector layout: {directory}')
    del vectors
    parquet = pq.ParquetFile(directory/'chips.geoparquet')
    if parquet.metadata.num_rows != run['chip_count']:
        raise ValueError(f'Vector/metadata row count differs: {directory}')
    offset = 0
    for batch in parquet.iter_batches(batch_size=4096,columns=['embedding_id','run_fingerprint']):
        if (not np.array_equal(batch.column(0).to_numpy(),np.arange(offset,offset+batch.num_rows))
            or any(value != run_fingerprint for value in batch.column(1).to_pylist())):
            raise ValueError(f'Metadata row alignment or compatibility differs: {directory}')
        offset += batch.num_rows
    geo = json.loads(parquet.schema_arrow.metadata[b'geo'])
    if CRS.from_json_dict(geo['columns']['geometry']['crs']).to_epsg() != 4326:
        raise ValueError(f'Unexpected common-coordinate system: {directory}')
    if verify_hashes and (sha256(directory/'embeddings.f16.npy') != run['vectors_sha256']
                         or sha256(directory/'chips.geoparquet') != run['metadata_sha256']):
        raise ValueError(f'Embedding bundle checksum failed: {directory}')
    return run


def top_matches(vectors: np.ndarray, query: np.ndarray, top_k: int) -> list[tuple[int,float]]:
    """Bounded exact cosine ranking; normalized query/vector semantics are shared."""
    if vectors.ndim != 2 or query.ndim != 1 or vectors.shape[1] != query.shape[0]:
        raise ValueError('Image and query vector dimensions must agree')
    if top_k <= 0 or not np.isfinite(query).all() or not np.isclose(np.linalg.norm(query),1,atol=.002):
        raise ValueError('Require a positive top-k and normalized finite query')
    candidates = []
    for start in range(0,len(vectors),4096):
        block = vectors[start:start+4096].astype(np.float32)
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite image embeddings')
        scores = block @ query
        count = min(top_k,len(scores))
        indices = np.argsort(-scores,kind='stable')[:count]
        candidates.extend((start+int(i),float(scores[i])) for i in indices)
        candidates = sorted(candidates,key=lambda item:(-item[1],item[0]))[:top_k]
    return candidates


def selected_rows(metadata_path: Path, row_ids: list[int]) -> dict[int,dict]:
    """Read only row groups containing top matches, not the statewide catalog."""
    wanted = set(row_ids)
    rows = {}
    offset = 0
    parquet = pq.ParquetFile(metadata_path)
    for index in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(index).num_rows
        selected = [row_id for row_id in wanted if offset <= row_id < offset+count]
        if selected:
            table = parquet.read_row_group(index).take(pa.array([i-offset for i in selected]))
            for row in table.to_pylist():
                rows[int(row['embedding_id'])] = row
        offset += count
    if set(rows) != wanted:
        raise ValueError('A search result does not have a matching metadata row')
    return rows
