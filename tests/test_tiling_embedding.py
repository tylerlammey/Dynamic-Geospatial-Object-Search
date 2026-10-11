"""NYS grid, RGB, common geometry, atomic bundle, and cosine search contracts."""
import json
from pathlib import Path
from queue import Queue
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely import from_wkb

from geospatial_search import bundles
from geospatial_search.download_nys_naip import Budget
from geospatial_search.download_nys_naip import sha256
from geospatial_search import embed_georsclip as embedding
from geospatial_search.embed_georsclip import Failure, prepare_source, duration, progress_metrics
from geospatial_search.model import normalized, load_config
from geospatial_search.search_text import read_collection, search_collection
from geospatial_search.tiling import Grid, Footprints, Source, read_rgb


def row(index,source='utm18/test.tif',run='run'):
    return {'embedding_id':index,'dataset_id':'nys','chip_id':f'{source}/{index}',
            'source_uri':source,'source_tile_id':'test','source_sha256':'source',
            'pixel_x':index,'pixel_y':0,'pixel_width':256,'pixel_height':256,
            'west':-74.,'south':42.,'east':-73.,'north':43.,'run_fingerprint':run,
            'geometry':__import__('shapely').to_wkb(__import__('shapely').geometry.box(-74,42,-73,43))}


class GridTests(unittest.TestCase):
    def test_explicit_progress_runtime_and_eta(self):
        metrics=progress_metrics(1000,250,3661,61,10,'running')
        self.assertEqual(metrics['percent_complete'],25)
        self.assertEqual(metrics['elapsed_hms'],'01:01:01')
        self.assertEqual(metrics['session_elapsed_seconds'],61)
        self.assertEqual(metrics['eta_seconds'],75)
        self.assertEqual(metrics['eta_hms'],'00:01:15')
        self.assertEqual(duration(90061),'25:01:01')
        self.assertIsNone(progress_metrics(1000,0,1,1,None,'running')['eta_seconds'])
        self.assertIsNone(progress_metrics(1000,250,10,10,10,'paused')['eta_seconds'])
        self.assertEqual(progress_metrics(1000,1000,10,10,10,'complete')['eta_seconds'],0)
        with self.assertRaises(ValueError):
            progress_metrics(100,101,0,0,None,'running')
    def test_edges_counts_and_no_gaps(self):
        for size,stride in [(256,192),(256,128),(512,256),(512,512)]:
            grid=Grid(size,stride)
            for length in (0,size-1,size,size+1,size+stride,size+stride+1,size+2137):
                starts=list(grid.starts(length))
                self.assertEqual(len(starts),grid.axis_count(length))
                self.assertEqual(starts,sorted(set(starts)))
                if starts:
                    self.assertEqual(starts[0],0)
                    self.assertEqual(starts[-1]+size,length)
                    self.assertTrue(all(b-a<=size for a,b in zip(starts,starts[1:])))
                    self.assertTrue(all(0<=x and x+size<=length for x in starts))
            self.assertEqual(grid.count(999,777),len(list(grid.windows(999,777))))

    def test_invalid_grid(self):
        for args in [(0,0),(-1,1),(256,257),(256,0),(256,1.5)]:
            with self.assertRaises(ValueError):
                Grid(*args)
        with self.assertRaises(ValueError):
            Grid(min_valid_fraction=1.1)

    def test_rgb_and_wgs84_in_three_zones(self):
        pixels=np.zeros((4,64,64),dtype=np.uint8)
        pixels[0]=25; pixels[1]=100; pixels[2]=180; pixels[3]=240
        for zone in (17,18,19):
            with MemoryFile() as memory:
                with memory.open(driver='GTiff',width=64,height=64,count=4,dtype='uint8',
                                 crs=f'EPSG:269{zone}',transform=from_origin(500000,4700000,.6,.6)) as writer:
                    writer.write(pixels)
                with memory.open() as dataset:
                    window=next(Grid(32,16).windows(64,64))
                    np.testing.assert_array_equal(np.asarray(read_rgb(dataset,window))[0,0],[25,100,180])
                    metadata=Footprints(dataset,32).metadata(window)
                    geometry=from_wkb(metadata['geometry'])
                    self.assertTrue(geometry.is_valid)
                    self.assertEqual(len(geometry.exterior.coords),37)
                    self.assertTrue(metadata['west']<metadata['centroid_lon']<metadata['east'])
                    self.assertTrue(metadata['south']<metadata['centroid_lat']<metadata['north'])
                    self.assertAlmostEqual(metadata['native_maxx']-metadata['native_minx'],19.2)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='nys-bundle-test-')
        self.root=Path(self.temporary.name)
        self.budget=Budget(self.root,cap=10_000_000,reserve=1000)

    def tearDown(self):
        self.temporary.cleanup()

    def publish(self,name,features,rows):
        return bundles.publish_bundle(self.budget,self.root/'shards'/name,features,rows,
            {'run_fingerprint':'run','embedding_dim':features.shape[1],'source_uri':f'utm18/{name}.tif'})

    def test_normalization_rejects_invalid_features(self):
        np.testing.assert_allclose(normalized(np.array([[3,4]],dtype=np.float16)),[[.6,.8]])
        for features in ([[0,0]],[[float('nan'),1]],[[float('inf'),1]]):
            with self.assertRaises(ValueError):
                normalized(np.array(features))

    def test_atomic_publish_resume_and_checksum(self):
        features=normalized(np.array([[3,4],[4,3]]))
        self.publish('test',features,[row(0),row(1)])
        directory=self.root/'shards/test'
        self.assertFalse((self.root/'shards/.test.staging').exists())
        self.assertEqual(bundles.validate_bundle(directory,'run',verify_hashes=True)['chip_count'],2)
        self.assertEqual(self.budget.total,sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()))
        with self.assertRaises(FileExistsError):
            self.publish('test',features,[row(0),row(1)])
        with self.assertRaises(ValueError):
            bundles.validate_bundle(directory,'other-run')
        vector=directory/'embeddings.f16.npy'
        with vector.open('r+b') as handle:
            handle.seek(-1,2); value=handle.read(1); handle.seek(-1,2); handle.write(bytes([value[0]^1]))
        with self.assertRaises(ValueError):
            bundles.validate_bundle(directory,'run',verify_hashes=True)

    def test_alignment_rejected_before_publication(self):
        with self.assertRaises(ValueError):
            self.publish('bad',normalized(np.array([[1,0],[0,1]])),[row(1),row(0)])
        self.assertFalse((self.root/'shards/bad').exists())

    def test_empty_source_can_be_atomically_recorded(self):
        self.publish('empty',np.empty((0,2),dtype=np.float32),[])
        self.assertEqual(bundles.validate_bundle(self.root/'shards/empty','run',verify_hashes=True)['chip_count'],0)

    def test_interrupted_stage_is_discarded_without_touching_completed_bundle(self):
        self.publish('test',normalized(np.array([[1,0]])),[row(0)])
        stage=self.root/'shards/.interrupted.staging'
        self.budget.write_bytes(stage/'embeddings.f16.npy',b'partial')
        bundles.discard_staging(self.budget,self.root/'shards')
        self.assertFalse(stage.exists())
        bundles.validate_bundle(self.root/'shards/test','run',verify_hashes=True)

    def test_sharded_ranking_and_metadata_ids_are_aligned(self):
        self.publish('a',normalized(np.array([[1,0],[0,1]])),[row(0,'utm18/a.tif'),row(1,'utm18/a.tif')])
        self.publish('b',normalized(np.array([[1,1],[-1,0]])),[row(0,'utm18/b.tif'),row(1,'utm18/b.tif')])
        collection={'run_fingerprint':'run','chip_count':4,'shards':[
            {'shard_id':'a','source_uri':'utm18/a.tif','chip_count':2},
            {'shard_id':'b','source_uri':'utm18/b.tif','chip_count':2}]}
        matches=search_collection(self.root,collection,np.array([1,0],dtype=np.float32),3)
        self.assertEqual([(s,i) for s,i,_ in matches],[('a',0),('b',0),('a',1)])
        selected=bundles.selected_rows(self.root/'shards/a/chips.geoparquet',[1])
        self.assertEqual(selected[1]['chip_id'],'utm18/a.tif/1')
        geo=json.loads(pq.ParquetFile(self.root/'shards/a/chips.geoparquet').schema_arrow.metadata[b'geo'])
        self.assertEqual(geo['columns']['geometry']['crs']['id']['code'],4326)

    def test_partial_and_mixed_recipe_collection_is_rejected(self):
        (self.root/'run.json').write_text(json.dumps({'run_fingerprint':'run'}))
        (self.root/'collection.json').write_text(json.dumps({'run_fingerprint':'other','state':'complete','shards':[]}))
        with self.assertRaises(ValueError):
            read_collection(self.root)
        (self.root/'collection.json').write_text(json.dumps({'run_fingerprint':'run','state':'running','chip_count':0,'shards':[]}))
        with self.assertRaises(ValueError):
            read_collection(self.root)
        self.assertEqual(read_collection(self.root,True)[1]['state'],'running')


class ReaderAndResumeTests(unittest.TestCase):
    def setUp(self):
        import rasterio
        self.temporary = tempfile.TemporaryDirectory(prefix='nys-reader-test-')
        self.root = Path(self.temporary.name)
        self.sources = []
        for name in ('a','b'):
            path = self.root/'imagery/utm18'/f'{name}.tif'
            path.parent.mkdir(parents=True,exist_ok=True)
            transform = from_origin(500000,4700000,.6,.6)
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(path,'w',driver='GTiff',width=64,height=64,count=3,dtype='uint8',
                                   crs='EPSG:26918',transform=transform) as writer:
                    writer.write(np.full((3,64,64),100,dtype=np.uint8))
                    mask=np.full((64,64),255,dtype=np.uint8)
                    mask[:16,:16]=0
                    writer.write_mask(mask)
            self.sources.append(Source(path,f'utm18/{name}.tif',64,64,'EPSG:26918',tuple(transform)[:6],
                                       sha256(path),path.stat().st_size,83))
        self.config=load_config()
        self.config.update(chip_size=32,stride=16,batch_size=2,workers=2)
        self.contract={name:self.config[name] for name in ('model_id','model_filename','model_revision','arch','base_pretrained','amp')}
        self.contract.update(model_fingerprint='test',preprocess_fingerprint='test',embedding_dim=2,
                             embedding_dtype='float16',metric='cosine',gpu='mock GPU')

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def preprocess(image):
        import torch
        return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255

    def test_reader_mask_filtering_and_checksum_failure(self):
        queue=Queue()
        with patch('torch.cuda.is_available',return_value=False):
            prepare_source(self.sources[0],Grid(32,16),self.config,self.preprocess,queue,threading.Event(),'run',self.contract)
        batches=[]
        while not queue.empty():
            item=queue.get_nowait()
            if item[0]=='batch':
                batches.extend(item[2])
            else:
                self.assertEqual(item,('done',9,1))
        self.assertEqual(len(batches),8)
        self.assertTrue(all(row['valid_fraction']==1 for row in batches))
        self.assertNotIn((0,0),[(r['pixel_x'],r['pixel_y']) for r in batches])
        with self.sources[0].path.open('ab') as handle:
            handle.write(b'changed')
        prepare_source(self.sources[0],Grid(32,16),self.config,self.preprocess,queue,threading.Event(),'run',self.contract)
        self.assertIsInstance(queue.get_nowait(),Failure)

    def test_failed_run_resumes_completed_source_without_reembedding(self):
        output=self.root/'embeddings/test'
        arguments=['--job-root',str(self.root),'--output-dir',str(output)]
        def vectors(model,inputs,device,amp):
            return np.tile(np.array([[1,0]],dtype=np.float32),(len(inputs),1))
        original_publish=embedding.publish_bundle
        def interrupted_publish(budget,directory,features,rows,run):
            if directory.name=='b':
                raise RuntimeError('simulated interruption')
            return original_publish(budget,directory,features,rows,run)
        with patch.object(embedding,'load_config',return_value=self.config), \
             patch.object(embedding,'load_inventory',return_value=(self.sources,'manifest')), \
             patch.object(embedding,'load_model',return_value=(object(),self.preprocess,'mock',self.contract)), \
             patch('torch.cuda.is_available',return_value=False), \
             patch.object(embedding,'encode_images',side_effect=vectors) as encode:
            with patch.object(embedding,'publish_bundle',side_effect=interrupted_publish):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    embedding.main(arguments)
            self.assertEqual(json.loads((output/'collection.json').read_text())['completed_sources'],1)
            failed_status=json.loads((output/'status.json').read_text())
            self.assertEqual(failed_status['durable_embedded_chips'],8)
            self.assertGreater(failed_status['elapsed_seconds'],0)
            first_hash=sha256(output/'shards/a/embeddings.f16.npy')
            encode.reset_mock()
            embedding.main(arguments+['--resume'])
            self.assertEqual(encode.call_count,4)
            self.assertEqual(first_hash,sha256(output/'shards/a/embeddings.f16.npy'))
            _,collection=read_collection(output)
            self.assertEqual(collection['chip_count'],16)
            self.assertEqual(collection['completed_sources'],2)
            self.assertEqual(json.loads((output/'status.json').read_text())['state'],'complete')
            final_status=json.loads((output/'status.json').read_text())
            self.assertEqual(final_status['percent_complete'],100)
            self.assertEqual(final_status['eta_seconds'],0)
            self.assertGreater(final_status['elapsed_seconds'],failed_status['elapsed_seconds'])
            self.assertEqual(final_status['durable_embedded_chips'],16)

    def test_graceful_pause_finishes_current_tile_searches_and_resumes(self):
        from geospatial_search import search_text
        output=self.root/'embeddings/pause-test'
        arguments=['--job-root',str(self.root),'--output-dir',str(output)]
        calls=[]
        def vectors(model,inputs,device,amp):
            calls.append(len(inputs))
            if len(calls)==1:
                (output/'PAUSE').touch()
            return np.tile(np.array([[1,0]],dtype=np.float32),(len(inputs),1))
        with patch.object(embedding,'load_config',return_value=self.config), \
             patch.object(embedding,'load_inventory',return_value=(self.sources,'manifest')), \
             patch.object(embedding,'load_model',return_value=(object(),self.preprocess,'mock',self.contract)), \
             patch('torch.cuda.is_available',return_value=False), \
             patch.object(embedding,'encode_images',side_effect=vectors):
            self.assertEqual(embedding.main(arguments),0)
            self.assertEqual(len(calls),4)  # Finish all eight accepted chips of tile A.
            root,collection=read_collection(output)
            self.assertEqual(collection['state'],'paused')
            self.assertEqual(collection['completed_sources'],1)
            self.assertEqual(collection['chip_count'],8)
            hashes={name:sha256(output/'shards/a'/name) for name in bundles.BUNDLE_FILES}
            with patch.object(search_text,'load_model',return_value=(object(),None,'mock',self.contract)), \
                 patch.object(search_text,'encode_text',return_value=np.array([1,0],dtype=np.float32)):
                self.assertEqual(search_text.main(['--query','test','--collection-dir',str(output),
                    '--imagery-dir',str(self.root/'imagery'),'--output-dir',str(self.root/'search-results'),'--top-k','3']),0)
            results=json.loads(next((self.root/'search-results').glob('*/results.json')).read_text())
            self.assertEqual(results['searched_chip_count'],8)
            self.assertTrue(results['partial_collection'])
            self.assertEqual(len(results['results']),3)
            calls.clear()
            # Avoid requesting another pause during resume.
            with patch.object(embedding,'encode_images',side_effect=lambda m,x,d,a:np.tile(np.array([[1,0]],dtype=np.float32),(len(x),1))) as encoder:
                self.assertEqual(embedding.main(arguments+['--resume']),0)
                self.assertEqual(encoder.call_count,4)
            self.assertFalse((output/'PAUSE').exists())
            self.assertEqual(hashes,{name:sha256(output/'shards/a'/name) for name in bundles.BUNDLE_FILES})
            self.assertEqual(read_collection(output)[1]['chip_count'],16)

    def test_search_rejects_active_worker_before_loading_gpu(self):
        from geospatial_search.download_nys_naip import job_lock
        from geospatial_search import search_text
        output=self.root/'embeddings/busy'
        output.mkdir(parents=True)
        with job_lock(output),patch.object(search_text,'load_model') as loader:
            with self.assertRaisesRegex(RuntimeError,'Pause'):
                search_text.main(['--query','test','--collection-dir',str(output),
                    '--imagery-dir',str(self.root/'imagery'),'--output-dir',str(self.root/'search-results')])
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
