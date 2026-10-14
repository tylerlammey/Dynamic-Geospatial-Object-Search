"""Synthetic regressions. Fixtures and temporary artifacts remain in frontend/."""
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer

from imagery import Inventory, TileRenderer, tile_bounds, lonlat_tile
from storage import DiskCache, contained,frontend_ownership,atomic_json
from retrieval import validate_query, verify_model_contract, status_snapshot

HERE = Path(__file__).resolve().parent
(HERE / '.tmp').mkdir(exist_ok=True)


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=HERE / '.tmp')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def inventory(self, zone, mask=False, offset=0):
        job = self.root / f'job{zone}{offset}'
        (job / 'imagery' / f'utm{zone}').mkdir(parents=True)
        (job / 'records').mkdir()
        lon = {17: -78.5, 18: -74.0, 19: -72.0}[zone]
        x, y = Transformer.from_crs(4326, 26900 + zone, always_xy=True).transform(lon, 43.0)
        affine = from_origin(x - 307.2, y + 307.2, .6, .6)
        name = f'synthetic{zone}{offset}.tif'
        p = job / 'imagery' / f'utm{zone}' / name
        pixels = np.zeros((3, 1024, 1024), dtype='uint8')
        pixels[0] = 201; pixels[1] = 57; pixels[2] = 19
        with rasterio.Env(GDAL_PAM_ENABLED=False):
            with rasterio.open(p, 'w', driver='GTiff', width=1024, height=1024,
                               count=3, dtype='uint8', crs=f'EPSG:{26900+zone}', transform=affine) as ds:
                ds.write(pixels)
                valid = np.full((1024, 1024), 255, dtype='uint8')
                if mask: valid[:, :512] = 0
                ds.write_mask(valid)
        bounds = rasterio.warp.transform_bounds(26900 + zone, 4326, x-307.2, y-307.2, x+307.2, y+307.2, densify_pts=21)
        record = dict(output=f'imagery/utm{zone}/{name}', output_bytes=p.stat().st_size,
                      output_sha256='synthetic', width=1024, height=1024, crs=f'EPSG:{26900+zone}',
                      transform=list(affine)[:6], bounds_wgs84=list(bounds))
        (job / 'records' / f'{name}.json').write_text(json.dumps(record))
        (job / 'manifest.json').write_text(json.dumps({'tiles':[{'name':name}]}))
        return Inventory(job, strict=False), lon, p


class Alignment(Fixtures):
    def test_provider_partial_edges_preserve_opaque_overlapping_coverage(self):
        from overview import composite
        background=np.array([[[200]],[[20]],[[10]],[[255]]],dtype='uint8')
        foreground=np.array([[[20]],[[40]],[[200]],[[128]]],dtype='uint8')
        composite(background,foreground)
        self.assertEqual(int(background[3,0,0]),255)
        np.testing.assert_allclose(background[:3,0,0],[110,30,105],atol=1)
    def test_photographic_overview_averages_fine_detail_instead_of_sparse_points(self):
        inventory,lon,source=self.inventory(18)
        # Alternating native columns alias to a single color with point sampling.
        with rasterio.Env(GDAL_PAM_ENABLED=False):
            with rasterio.open(source,'r+') as ds:
                pixels=np.zeros((3,1024,1024),dtype='uint8');pixels[:,:,1::2]=255;ds.write(pixels)
        inventory=Inventory(inventory.job,strict=False)
        renderer=TileRenderer(inventory,DiskCache(self.root/'cache',5_000_000),workers=1)
        image=np.asarray(renderer.photograph(inventory.sources[0]))
        self.assertEqual(image.shape,(512,512,4))
        self.assertLess(image[20:-20,20:-20,:3].std(),1)
        self.assertAlmostEqual(float(image[20:-20,20:-20,:3].mean()),127.5,delta=1)
        self.assertEqual(int(image[256,256,3]),255)
        renderer.close()

    def test_prepared_state_tiles_do_not_reopen_originals(self):
        from overview import prepare_overview
        inventory,lon,source=self.inventory(19)
        cache=DiskCache(self.root/'cache',5_000_000)
        renderer=TileRenderer(inventory,cache,workers=1)
        updates=[];prepare_overview(renderer,updates.append,workers=1)
        xyz=lonlat_tile(lon,43,10)
        with patch.object(inventory,'check',side_effect=AssertionError('Original TIFF reopened')):
            image=np.asarray(Image.open(io.BytesIO(renderer.tile(*xyz))))
        west,south,east,north=tile_bounds(*xyz)
        mx,my=Transformer.from_crs(4326,3857,always_xy=True).transform(lon,43)
        col=int((mx-west)/(east-west)*256);row=int((north-my)/(north-south)*256)
        self.assertGreater(int(image[row,col,3]),0)
        np.testing.assert_allclose(image[row,col,:3],[201,57,19],atol=2)
        self.assertTrue(renderer.overview_ready())
        self.assertFalse(source.with_suffix('.tif.aux.xml').exists())
        self.assertEqual(updates[-1]['state'],'complete')
        renderer.close()

    def test_photographic_average_excludes_invalid_rgb_at_partial_edges(self):
        inventory,lon,source=self.inventory(18)
        pixels=np.zeros((3,1024,1024),dtype='uint8');pixels[:]=np.array([10,20,30])[:,None,None]
        pixels[:,:,513:]=np.array([240,50,200])[:,None,None]
        mask=np.full((1024,1024),255,dtype='uint8');mask[:,513:]=0
        with rasterio.Env(GDAL_PAM_ENABLED=False):
            with rasterio.open(source,'r+') as ds:ds.write(pixels);ds.write_mask(mask)
        record_path=inventory.job/'records'/f'{source.name}.json'
        record=json.loads(record_path.read_text());record['output_bytes']=source.stat().st_size;record_path.write_text(json.dumps(record))
        inventory=Inventory(inventory.job,strict=False)
        renderer=TileRenderer(inventory,DiskCache(self.root/'cache',5_000_000),workers=1)
        image=np.asarray(renderer.photograph(inventory.sources[0]))
        np.testing.assert_allclose(image[256,256,:3],[10,20,30],atol=1)
        self.assertTrue(120<=int(image[256,256,3])<=130)
        renderer.close()

    def test_cancelled_tile_does_not_read_sources(self):
        import threading
        from concurrent.futures import CancelledError
        inventory,lon,_=self.inventory(17)
        renderer=TileRenderer(inventory,DiskCache(self.root/'cache',1_000_000),workers=1)
        cancel=threading.Event();cancel.set()
        with patch.object(inventory,'check',side_effect=AssertionError('Cancelled request opened a source')):
            with self.assertRaises(CancelledError):renderer.tile(*lonlat_tile(lon,43,15),cancel=cancel)
        renderer.close()

    def test_rgb_and_web_mercator_alignment_in_all_utm_zones(self):
        for zone in (17, 18, 19):
            inventory, lon, source = self.inventory(zone)
            renderer = TileRenderer(inventory, DiskCache(self.root / f'cache{zone}', 1000000), workers=1)
            z, x, y = lonlat_tile(lon, 43, 18)
            image = np.asarray(Image.open(io.BytesIO(renderer.tile(z, x, y))))
            # The actual transformed source covers this tile's center; colors must stay RGB.
            self.assertEqual(image.shape, (256, 256, 4))
            self.assertGreater(image[128,128,3], 250)
            np.testing.assert_allclose(image[128,128,:3], [201,57,19], atol=1)
            self.assertFalse(source.with_suffix('.tif.aux.xml').exists())
            inspected = renderer.inspect(lon, 43)
            self.assertEqual(inspected['pixel_width'], 256)
            self.assertAlmostEqual(inspected['centroid_lon'], lon, places=4)
            preview = Image.open(io.BytesIO(renderer.preview(inspected)))
            self.assertEqual(preview.size, (256,256))
            renderer.close()

    def test_internal_mask_remains_transparent(self):
        inventory, lon, _ = self.inventory(18, mask=True)
        renderer = TileRenderer(inventory, DiskCache(self.root/'cache', 1000000), workers=1)
        west = inventory.sources[0].bounds[0] + .0001
        z,x,y = lonlat_tile(west,43,18)
        image = np.asarray(Image.open(io.BytesIO(renderer.tile(z,x,y))))
        self.assertEqual(image[128,128,3],0)
        renderer.close()

    def test_photographic_overview_uses_real_rgb_and_alpha(self):
        inventory, lon, _ = self.inventory(19)
        renderer = TileRenderer(inventory, DiskCache(self.root/'cache',1000000), workers=1)
        photograph = np.asarray(renderer.photograph(inventory.sources[0]))
        self.assertEqual(photograph.shape,(512,512,4))
        np.testing.assert_equal(photograph[128,128],[201,57,19,255])
        renderer.close()

    def test_xyz_round_trip_and_input_bounds(self):
        z,x,y=lonlat_tile(-74,43,15)
        west,south,east,north=tile_bounds(z,x,y,geographic=True)
        self.assertTrue(west<=-74<=east and south<=43<=north)
        with self.assertRaises(ValueError): tile_bounds(15,-1,2)


class CacheSafety(Fixtures):
    def test_atomic_json_retries_windows_locks_without_losing_previous_status(self):
        target=self.root/'status.json';atomic_json(target,{'state':'encoding'},self.root)
        replace=os.replace;attempts=[]
        def locked_then_replace(source,destination):
            attempts.append(1)
            if len(attempts)<3:
                self.assertEqual(json.loads(target.read_text()),{'state':'encoding'})
                error=PermissionError('Temporary Windows reader lock');error.winerror=5
                raise error
            return replace(source,destination)
        with patch('storage.os.replace',side_effect=locked_then_replace),patch('time.sleep'):
            atomic_json(target,{'state':'searching'},self.root)
        self.assertEqual(len(attempts),3)
        self.assertEqual(json.loads(target.read_text()),{'state':'searching'})
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_permanent_replace_failure_preserves_cache_accounting_and_cleans_stage(self):
        cache=DiskCache(self.root/'cache',100);cache.write('a',b'original')
        before=cache.stats()
        error=PermissionError('Persistent Windows denial');error.winerror=5
        with patch('storage.os.replace',side_effect=error) as replace,patch('time.sleep'):
            with self.assertRaises(PermissionError):cache.write('a',b'replacement')
        self.assertGreater(replace.call_count,1)
        self.assertLessEqual(replace.call_count,40)
        self.assertEqual(cache.stats(),before)
        self.assertEqual(cache.read('a'),b'original')
        self.assertFalse(list(cache.root.glob('*.tmp')))

    def test_unrelated_replace_error_is_not_retried(self):
        target=self.root/'status.json';atomic_json(target,{'state':'encoding'},self.root)
        with patch('storage.os.replace',side_effect=OSError('Disk failure')) as replace,patch('time.sleep') as sleep:
            with self.assertRaises(OSError):atomic_json(target,{'state':'searching'},self.root)
        self.assertEqual(replace.call_count,1);sleep.assert_not_called()
        self.assertEqual(json.loads(target.read_text()),{'state':'encoding'})
        self.assertFalse(list(self.root.glob('*.tmp')))

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_live_windows_reader_lock_can_release_during_atomic_status_write(self):
        target=self.root/'status.json';atomic_json(target,{'state':'encoding'},self.root)
        retry=threading.Event();finished=threading.Event();errors=[];original_sleep=time.sleep
        def observed_sleep(seconds):retry.set();original_sleep(seconds)
        def writer():
            try:atomic_json(target,{'state':'searching'},self.root)
            except Exception as exc:errors.append(exc)
            finally:finished.set()
        thread=threading.Thread(target=writer)
        with patch('time.sleep',side_effect=observed_sleep):
            try:
                with target.open('rb') as reader:
                    thread.start()
                    self.assertTrue(retry.wait(2),'The open reader must force a Windows replace retry')
                    self.assertFalse(finished.is_set())
                    self.assertEqual(json.load(reader),{'state':'encoding'})
            finally:
                thread.join(7)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        self.assertEqual(json.loads(target.read_text()),{'state':'searching'})
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_staging_bytes_are_counted_and_frontend_ownership_is_exclusive(self):
        cache=DiskCache(self.root/'cache',100);observed=[]
        def measure(_):observed.append(sum(p.stat().st_size for p in cache.root.rglob('*') if p.is_file()))
        with patch('storage.os.fsync',side_effect=measure):
            cache.write('a',b'a'*45);cache.write('b',b'b'*45);cache.write('a',b'c'*60)
        self.assertTrue(observed)
        self.assertLessEqual(max(observed),100)
        with frontend_ownership(self.root):
            with self.assertRaises(RuntimeError):
                with frontend_ownership(self.root):pass
        with frontend_ownership(self.root):pass

    def test_atomic_writes_never_exceed_cap_and_evict_lru(self):
        cache=DiskCache(self.root/'cache',100)
        cache.write('a',b'a'*40);cache.write('b',b'b'*40)
        cache.read('a')
        cache.write('c',b'c'*50)
        self.assertLessEqual(cache.stats()['bytes'],100)
        self.assertIsNone(cache.read('b'))
        self.assertEqual(cache.read('a'),b'a'*40)
        with self.assertRaises(ValueError):cache.write('too-large',b'x'*101)
        self.assertFalse(any(p.suffix=='.tmp' for p in cache.root.rglob('*')))

    def test_clear_and_traversal_cannot_touch_sources(self):
        outside=self.root/'imagery.tif';outside.write_bytes(b'original')
        cache=DiskCache(self.root/'cache',100)
        cache.write('a',b'preview')
        with self.assertRaises(ValueError):cache.write('../imagery.tif',b'bad')
        with self.assertRaises(ValueError):contained(cache.root,outside)
        cache.clear()
        self.assertEqual(outside.read_bytes(),b'original')
        self.assertEqual(cache.stats()['bytes'],0)


class SearchContract(Fixtures):
    def test_adapter_exact_ranking_snapshot_metadata_and_contract_checks(self):
        from geospatial_search.bundles import publish_bundle
        from geospatial_search.download_nys_naip import Budget,sha256
        from geospatial_search.tiling import Footprints
        from geospatial_search import search_text
        from retrieval import ExactRetrieval
        inventory,lon,source=self.inventory(18)
        frontend=self.root/'frontend';frontend.mkdir()
        collection=inventory.job/'embeddings/georsclip-vith14-s256-st192';collection.mkdir(parents=True)
        uri=inventory.sources[0].uri;source_hash=sha256(source)
        rows=[]
        with rasterio.Env(GDAL_PAM_ENABLED=False):
            with rasterio.open(source) as ds:
                for index in range(3):
                    meta=Footprints(ds,256).metadata(rasterio.windows.Window(index*256,256,256,256))
                    rows.append({'embedding_id':index,'chip_id':f'synthetic-{index}','dataset_id':'synthetic','source_uri':uri,'source_tile_id':'synthetic','source_sha256':source_hash,'pixel_x':index*256,'pixel_y':256,'pixel_width':256,'pixel_height':256,'run_fingerprint':'synthetic-run','scale_id':'native060cm-s256-st192',**meta})
        published=publish_bundle(Budget(collection,cap=10_000_000,reserve=0),collection/'shards'/'synthetic',np.array([[1,0],[.8,.6],[0,1]],dtype=np.float32),rows,{'run_fingerprint':'synthetic-run','embedding_dim':2,'source_uri':uri})
        contract={name:'synthetic' for name in ('model_id','model_revision','arch','model_fingerprint','preprocess_fingerprint')};contract.update(embedding_dim=2,amp=False)
        (collection/'run.json').write_text(json.dumps({'run_fingerprint':'synthetic-run','model_contract':contract,'recipe':{'model':{'arch':'synthetic'}}}))
        catalog={'run_fingerprint':'synthetic-run','state':'paused','chip_count':3,'total_sources':3805,'shards':[{'shard_id':'synthetic','chip_count':3,'source_uri':uri}]}
        (collection/'collection.json').write_text(json.dumps(catalog))
        adapter=ExactRetrieval(self.root,frontend);adapter.job=inventory.job;adapter.collection=collection
        updates=[];original_tqdm=search_text.tqdm
        try:
            with patch('geospatial_search.model.load_model',return_value=(object(),None,'synthetic',contract)) as loader,patch('geospatial_search.model.encode_text',return_value=np.array([1,0],dtype=np.float32)):
                result=adapter.text({'query':'test','top_k':3},updates.append)
                self.assertIs(search_text.tqdm,original_tqdm,'Frontend retrieval must not replace CLI progress globally')
                self.assertEqual([p for p in updates if p['state'] in ('searching','metadata')],
                    [{'searched_source_tiles':1,'total_source_tiles':3805,'state':'searching','scanned_sources':0},
                     {'searched_source_tiles':1,'total_source_tiles':3805,'state':'metadata','scanned_sources':1}])
                self.assertEqual([r['embedding_id'] for r in result['results']],[0,1,2])
                self.assertAlmostEqual(result['results'][1]['score'],.8,places=3)
                self.assertEqual(result['searched_source_tiles'],1)
                self.assertEqual(result['searched_source_uris'],[uri])
                self.assertEqual(result['results'][0]['geometry']['type'],'Polygon')
                self.assertEqual(result['results'][0]['pixel_x'],0)
                adapter.text({'query':'test','top_k':2},updates.append)
                self.assertEqual(loader.call_count,1,'Repeat searches must reuse the fingerprinted CPU text vector')
                with self.assertRaises(ValueError):
                    with patch('geospatial_search.model.load_model',return_value=(object(),None,'synthetic',{**contract,'model_fingerprint':'different'})):
                        adapter.text({'query':'different','top_k':1},updates.append)
                adapter.preload(progress=updates.append)
                self.assertEqual(loader.call_count,2)
                self.assertTrue(status_snapshot(collection,inventory.job)['busy'],'The resident model must hold both backend locks')
                try:
                    first=adapter.text({'query':'new resident phrase one','top_k':3},updates.append)
                    second=adapter.text({'query':'new resident phrase two','top_k':3},updates.append)
                    self.assertEqual(loader.call_count,2,'Distinct queries must reuse the already loaded GPU model')
                    self.assertEqual(first['results'],second['results'])
                    changed={**catalog,'run_fingerprint':'other-run'}
                    (collection/'collection.json').write_text(json.dumps(changed))
                    (collection/'run.json').write_text(json.dumps({'run_fingerprint':'other-run','model_contract':contract,'recipe':{'model':{'arch':'synthetic'}}}))
                    with self.assertRaisesRegex(ValueError,'another collection fingerprint'):
                        adapter.text({'query':'changed collection'},updates.append)
                    (collection/'collection.json').write_text(json.dumps(catalog))
                    (collection/'run.json').write_text(json.dumps({'run_fingerprint':'synthetic-run','model_contract':contract,'recipe':{'model':{'arch':'synthetic'}}}))
                finally:adapter.close()
                self.assertFalse(status_snapshot(collection,inventory.job)['busy'],'Unloading must release collection and GPU locks')
                with patch('geospatial_search.model.load_model',return_value=(object(),None,'synthetic',{**contract,'model_fingerprint':'wrong'})):
                    with self.assertRaisesRegex(ValueError,'model_fingerprint'):adapter.preload()
                self.assertIsNone(adapter.resident_model)
                self.assertIsNone(adapter.resident_locks)
                self.assertFalse(status_snapshot(collection,inventory.job)['busy'],'Failed preload must release both locks')
            self.assertTrue(any(p['state']=='searching' for p in updates))
            self.assertEqual(json.loads((collection/'collection.json').read_text()),catalog)
        finally:self.assertIs(search_text.tqdm,original_tqdm,'Search progress renderer must remain unchanged even after failure')

    def test_queries_and_contract_mismatch_fail_closed(self):
        self.assertEqual(validate_query({'query':' baseball field ','top_k':25}),('baseball field',25,False))
        for invalid in [{'query':'','top_k':1},{'query':'jet','top_k':0},{'query':'x','top_k':True},{'query':'x','top_k':5001}]:
            with self.assertRaises(ValueError):validate_query(invalid)
        stored={name:'value' for name in ('model_id','model_revision','arch','model_fingerprint','preprocess_fingerprint','embedding_dim')}
        verify_model_contract(stored,stored)
        with self.assertRaises(ValueError):verify_model_contract(stored,{**stored,'model_revision':'different'})

    def test_partial_coverage_uses_published_catalog_not_live_counters(self):
        collection=self.root/'collection';collection.mkdir()
        (collection/'collection.json').write_text(json.dumps({'run_fingerprint':'same','state':'running','shards':[{'shard_id':'s','chip_count':11,'source_uri':'s.tif'}],'chip_count':11,'total_sources':3805}))
        (collection/'run.json').write_text(json.dumps({'run_fingerprint':'same'}))
        (collection/'status.json').write_text(json.dumps({'state':'running','completed_sources':999,'embedded_chips':10000}))
        with patch('retrieval.lock_held',return_value=True):
            status=status_snapshot(collection,self.root)
        self.assertEqual(status['published_sources'],1)
        self.assertEqual(status['published_chips'],11)
        self.assertTrue(status['busy'])
        self.assertTrue(status['partial'])


if __name__=='__main__': unittest.main()
