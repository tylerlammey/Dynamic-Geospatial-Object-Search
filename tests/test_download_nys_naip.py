"""Small local tests of quota, restart, and raster preservation contracts."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from geospatial_search import download_nys_naip as p


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix='nys-naip-test-')
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_default_output_is_ignored_repository_data_directory(self):
        expected = Path(__file__).resolve().parents[1] / 'data' / 'nys-naip-2022-rgb-060m'
        self.assertEqual(p.parse_args([]).output_dir, expected)

    def test_check_running_uses_selected_folder_without_creating_missing_job(self):
        missing = self.root / 'new-job'
        self.assertEqual(p.main(['--output-dir',str(missing),'--check-running']),1)
        self.assertFalse(missing.exists())
        with p.job_lock(self.root):
            self.assertEqual(p.main(['--output-dir',str(self.root),'--check-running']),0)
        self.assertEqual(p.main(['--output-dir',str(self.root),'--check-running']),1)

    def test_prepare_anchors_artifacts_and_scratch_to_selected_folder(self):
        selected = self.root / 'nested' / 'job'
        manifest = {'tile_count':0,'tiles':[]}
        def prepare(budget):
            self.assertEqual(budget.root, selected)
            self.assertEqual(p.os.environ['GDAL_TMPDIR'], str(selected/'temporary'))
            budget.atomic_json(selected/'manifest.json', manifest)
            return manifest
        with patch.dict(p.os.environ), patch.object(p,'prepare',side_effect=prepare):
            self.assertEqual(p.main(['--output-dir',str(selected),'--prepare']),0)
        self.assertEqual(json.loads((selected/'manifest.json').read_text()),manifest)
        self.assertTrue((selected/'imagery'/'utm17').is_dir())
        self.assertTrue((selected/'imagery'/'utm18').is_dir())
        self.assertTrue((selected/'imagery'/'utm19').is_dir())

    def test_rejected_write_does_not_grow_file(self):
        budget = p.Budget(self.root, cap=64, reserve=8)
        path = self.root / 'data.bin'
        budget.write_bytes(path, b'a' * 48)
        with self.assertRaises(p.BudgetExceeded):
            budget.write_bytes(path, b'b' * 9, append=True)
        self.assertEqual(path.read_bytes(), b'a' * 48)
        self.assertEqual(budget.total, 48)
        with self.assertRaises(ValueError):
            budget.write_bytes(self.root.parent / 'escape.bin', b'x')

    def test_atomic_checkpoint_counts_both_files_before_replace(self):
        budget = p.Budget(self.root, cap=512, reserve=8)
        path = self.root / 'status.json'
        budget.atomic_json(path, {'state': 'old'})
        budget.atomic_json(path, {'state': 'new', 'completed': 2})
        self.assertEqual(json.loads(path.read_text())['completed'], 2)
        self.assertEqual(budget.total, path.stat().st_size)
        self.assertGreater(budget.peak, budget.total)
        self.assertFalse(path.with_name('status.json.new').exists())

    def test_atomic_replace_retries_windows_sharing_conflicts(self):
        budget=p.Budget(self.root,cap=10000,reserve=100)
        path=self.root/'status.json'
        budget.atomic_json(path,{'state':'old'})
        replace=p.os.replace
        calls=0
        def temporarily_locked(source,destination):
            nonlocal calls
            calls+=1
            if calls<=3:
                raise PermissionError(13,'simulated sharing violation')
            return replace(source,destination)
        with patch.object(p.os,'replace',side_effect=temporarily_locked):
            with patch.object(p.time,'sleep') as sleep:
                budget.atomic_json(path,{'state':'new'})
        self.assertEqual(calls,4)
        self.assertEqual(sleep.call_count,3)
        self.assertEqual(json.loads(path.read_text())['state'],'new')
        self.assertEqual(budget.total,path.stat().st_size)

    def test_locked_status_is_nonfatal_but_provenance_is_mandatory(self):
        budget=p.Budget(self.root,cap=100000,reserve=1000)
        path=self.root/'status.json'
        budget.atomic_json(path,{'state':'old'})
        old_size=path.stat().st_size
        with patch.object(p.os,'replace',side_effect=PermissionError(13,'locked')):
            with patch.object(p.time,'sleep'):
                result=p.write_status(budget,{'raw_rgb_bytes':100,'tile_count':1},'running',{})
                self.assertFalse(result)
                self.assertEqual(json.loads(path.read_text())['state'],'old')
                self.assertEqual(path.stat().st_size,old_size)
                with self.assertRaises(PermissionError):
                    budget.atomic_json(self.root/'records/tile.json',{'complete':True})
        budget._status_retry_after=0
        self.assertTrue(p.write_status(budget,{'raw_rgb_bytes':100,'tile_count':1},'running',{}))
        self.assertEqual(json.loads(path.read_text())['state'],'running')
        self.assertEqual(budget.total,sum(path.stat().st_size for path in self.root.rglob('*') if path.is_file()))

    def test_declared_and_actual_download_lengths_are_bounded(self):
        budget = p.Budget(self.root, cap=128, reserve=8)
        response = io.BytesIO(b'a' * 100)
        response.headers = {'Content-Length': '100'}
        path = self.root / 'source.part'
        with patch.object(p.urllib.request, 'urlopen', return_value=response):
            with self.assertRaises(p.BudgetExceeded):
                p.download(p.BASE + 'test.tif', path, budget, 80, check_stop=False)
        self.assertFalse(path.exists())
        response = io.BytesIO(b'a' * 100)
        response.headers = {}
        with patch.object(p.urllib.request, 'urlopen', return_value=response):
            with self.assertRaises(p.BudgetExceeded):
                p.download(p.BASE + 'test.tif', path, budget, 80, check_stop=False)
        self.assertEqual(path.stat().st_size, 0)

    def test_missing_output_is_not_complete_and_corruption_is_rejected(self):
        budget = p.Budget(self.root, cap=10_000, reserve=100)
        entry = {'name': 'tile.tif', 'zone': 18, 'url': p.BASE + 'tile.tif'}
        record_path = self.root / 'records' / 'tile.tif.json'
        output = self.root / 'imagery' / 'utm18' / 'tile.tif'
        budget.atomic_json(record_path, {'source_url': entry['url'], 'jpeg_quality':83,
                                       'resolution_m_per_pixel':0.6, 'output_bytes':3,
                                       'output_sha256':p.hashlib.sha256(b'abc').hexdigest()})
        self.assertIsNone(p.verified_record(entry, budget))
        budget.write_bytes(output, b'abc')
        self.assertIsNotNone(p.verified_record(entry, budget))
        budget.write_bytes(output, b'abd')
        with self.assertRaises(ValueError):
            p.verified_record(entry, budget)

    def test_stop_prevents_work(self):
        (self.root / 'STOP').write_text('pause')
        with self.assertRaises(p.PauseRequested):
            p.check_pause(self.root)

    def test_concurrent_writes_share_one_atomic_budget(self):
        budget=p.Budget(self.root,cap=128,reserve=8)
        start=threading.Barrier(4)
        def write(index):
            start.wait()
            try:
                budget.write_bytes(self.root/f'{index}.bin',b'x'*50)
                return True
            except p.BudgetExceeded:
                return False
        with ThreadPoolExecutor(max_workers=4) as pool:
            result=list(pool.map(write,range(4)))
        self.assertEqual(sum(result),2)
        self.assertEqual(budget.total,100)
        self.assertEqual(sum(path.stat().st_size for path in self.root.glob('*.bin')),100)
        self.assertLessEqual(budget.peak,120)

    def test_concurrent_download_transactions_are_bounded(self):
        budget=p.Budget(self.root,cap=128,reserve=8)
        barrier=threading.Barrier(2)
        def response(*args,**kwargs):
            barrier.wait()
            stream=io.BytesIO(b'x'*80)
            stream.headers={'Content-Length':'80'}
            return stream
        def download(index):
            try:
                p.download(p.BASE+'test.tif',self.root/f'{index}.part',budget,80,check_stop=False)
                return True
            except p.BudgetExceeded:
                return False
        with patch.object(p.urllib.request,'urlopen',side_effect=response):
            with ThreadPoolExecutor(max_workers=2) as pool:
                result=list(pool.map(download,range(2)))
        self.assertEqual(sum(result),1)
        self.assertEqual(budget.total,80)
        self.assertEqual(sum(path.stat().st_size for path in self.root.glob('*.part')),80)

    def test_parallel_eta_uses_completion_intervals_and_current_session(self):
        completed={
            'old':{'raw_rgb_bytes':100,'elapsed_seconds':100,'verified_utc':'a'},
            'first':{'raw_rgb_bytes':100,'elapsed_seconds':10,'throughput_seconds':10,
                     'throughput_epoch':'new','verified_utc':'b'},
            'second':{'raw_rgb_bytes':100,'elapsed_seconds':10,'throughput_seconds':.1,
                      'throughput_epoch':'new','verified_utc':'c'}}
        estimate=p.estimate_progress({'raw_rgb_bytes':500,'tile_count':5},completed)
        self.assertEqual(estimate['eta_seconds'],11)
        self.assertEqual(estimate['eta_sample_tiles'],2)

    def test_parallel_eta_waits_for_first_wave_then_excludes_fill(self):
        records={}
        manifest={'raw_rgb_bytes':2000,'tile_count':20}
        for index in range(6):
            records[str(index)]={'raw_rgb_bytes':100,'elapsed_seconds':10,'verified_utc':str(index),
                                'throughput_seconds':20 if index==0 else 2,
                                'execution_workers':2,'throughput_epoch':'session'}
            if index==0:
                self.assertIsNone(p.estimate_progress(manifest,records)['eta_seconds'])
        estimate=p.estimate_progress(manifest,records)
        self.assertEqual(estimate['eta_seconds'],28)
        self.assertEqual(estimate['eta_sample_tiles'],4)

    def test_parallel_scheduler_has_bounded_active_tiles_and_durable_records(self):
        budget=p.Budget(self.root,cap=100_000,reserve=1000)
        entries=[{'name':f'tile{i}.tif'} for i in range(6)]
        progress=p.Progress(budget,{'raw_rgb_bytes':600,'tile_count':6},{})
        progress.workers=2
        active=0; peak=0; lock=threading.Lock()
        def work(entry,budget,progress):
            nonlocal active,peak
            with lock:
                active+=1; peak=max(peak,active)
            progress.worker(entry['name'])('downloading',1,2,force=True)
            time.sleep(.025)
            with lock:
                active-=1
            return {'raw_rgb_bytes':100,'output_bytes':10,'elapsed_seconds':.025,'verified_utc':p.now()}
        with patch.object(p,'tile_with_retries',side_effect=work):
            p.run_tiles(entries,budget,progress,2)
        self.assertEqual(peak,2)
        self.assertEqual(len(progress.completed),6)
        self.assertFalse(progress.active)
        for entry in entries:
            record=json.loads((self.root/'records'/(entry['name']+'.json')).read_text())
            self.assertEqual(record['execution_workers'],2)
            self.assertIn('throughput_seconds',record)
        self.assertEqual(budget.total,sum(path.stat().st_size for path in self.root.rglob('*') if path.is_file()))

    def test_parallel_scheduler_retries_transient_errors(self):
        budget=p.Budget(self.root,cap=100_000,reserve=1000)
        progress=p.Progress(budget,{'raw_rgb_bytes':100,'tile_count':1},{})
        expected={'raw_rgb_bytes':100,'output_bytes':10,'elapsed_seconds':1}
        with patch.object(p,'process_tile',side_effect=[OSError('transient'),expected]) as work:
            with patch.object(p.time,'sleep'):
                actual=p.tile_with_retries({'name':'a.tif'},budget,progress)
        self.assertEqual(actual,expected)
        self.assertEqual(work.call_count,2)

    def test_partial_cleanup_only_removes_known_job_owned_names(self):
        budget=p.Budget(self.root,cap=10000,reserve=100)
        temporary=self.root/'temporary'; temporary.mkdir()
        owned=temporary/'m_4207318_se_18_060_20220731.tif.source.part'
        keep=temporary/'unrelated.txt'
        budget.write_bytes(owned,b'partial'); budget.write_bytes(keep,b'keep')
        p.cleanup_temporary(budget)
        self.assertFalse(owned.exists())
        self.assertEqual(keep.read_bytes(),b'keep')
        self.assertEqual(budget.total,4)

    def test_two_real_rasters_process_concurrently_and_resume_by_hash(self):
        pixels=np.random.default_rng(3).integers(10,240,(4,259,257),dtype=np.uint8)
        pixels[:,:17,:13]=0
        transform=from_origin(600000,4700000,.6,.6)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True,GDAL_PAM_ENABLED=False):
            with MemoryFile() as memory:
                with memory.open(driver='GTiff',width=257,height=259,count=4,dtype='uint8',
                                 crs='EPSG:26918',transform=transform,nodata=0) as source:
                    source.write(pixels)
                    source.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.undefined)
                blob=memory.read()
                with memory.open() as source:
                    expected_mask=source.dataset_mask()
        budget=p.Budget(self.root,cap=2_000_000,reserve=10000)
        entries=[]
        for quadrant in ('ne','nw'):
            name=f'm_4207318_{quadrant}_18_060_20220731.tif'
            entries.append(dict(name=name,url=p.BASE+name,zone=18,width=257,height=259,
                                crs='EPSG:26918',transform=list(transform)[:6],listed_bytes=len(blob),
                                raw_rgb_bytes=257*259*3,max_output_bytes=1_000_000))
        progress=p.Progress(budget,{'raw_rgb_bytes':2*257*259*3,'tile_count':2},{})
        progress.workers=2
        def response(*args,**kwargs):
            stream=io.BytesIO(blob); stream.headers={'Content-Length':str(len(blob))}
            return stream
        with patch.object(p.urllib.request,'urlopen',side_effect=response):
            p.run_tiles(entries,budget,progress,2)
        self.assertEqual(len(progress.completed),2)
        for entry in entries:
            record=p.verified_record(entry,budget)
            self.assertEqual(record['source_download']['sha256'],p.hashlib.sha256(blob).hexdigest())
            with rasterio.open(self.root/record['output']) as output:
                self.assertEqual(output.transform,transform)
                self.assertEqual(output.count,3)
                np.testing.assert_array_equal(output.dataset_mask(),expected_mask)
        self.assertFalse(list((self.root/'temporary').glob('*.part')))
        self.assertEqual(budget.total,sum(path.stat().st_size for path in self.root.rglob('*') if path.is_file()))

    def test_lock_rejects_second_worker_and_releases_after_exit(self):
        with p.job_lock(self.root):
            with self.assertRaises(RuntimeError):
                with p.job_lock(self.root):
                    self.fail('Second worker acquired the lock')
        with p.job_lock(self.root):
            pass

    def test_eta_accounts_for_remaining_pixels_and_unknown_speed(self):
        manifest = {'raw_rgb_bytes':1000, 'tile_count':4}
        self.assertIsNone(p.estimate_progress(manifest,{})['eta_seconds'])
        completed = {
            'a':{'raw_rgb_bytes':100,'elapsed_seconds':10,'verified_utc':'2026-10-10T10:00:00Z'},
            'b':{'raw_rgb_bytes':200,'elapsed_seconds':20,'verified_utc':'2026-10-10T10:01:00Z'}}
        estimate = p.estimate_progress(manifest,completed)
        self.assertEqual(estimate['eta_seconds'],70)
        self.assertEqual(estimate['percent_complete'],50)
        self.assertEqual(estimate['eta_sample_tiles'],2)
        completed['c']={'raw_rgb_bytes':300,'elapsed_seconds':30,'verified_utc':'2026-10-10T10:02:00Z'}
        completed['d']={'raw_rgb_bytes':400,'elapsed_seconds':40,'verified_utc':'2026-10-10T10:03:00Z'}
        self.assertEqual(p.estimate_progress(manifest,completed)['eta_seconds'],0)

    def test_eta_uses_recent_speed_instead_of_old_history(self):
        records = {str(i):{'raw_rgb_bytes':100,'elapsed_seconds':10 if i else 1000,
                           'verified_utc':f'{i:03d}'} for i in range(31)}
        estimate = p.estimate_progress({'raw_rgb_bytes':4000,'tile_count':40},records)
        self.assertEqual(estimate['eta_seconds'],90)
        self.assertEqual(estimate['eta_sample_tiles'],30)

    def test_status_contains_phase_eta_and_preserves_peak_on_restart(self):
        budget = p.Budget(self.root, cap=100_000, reserve=1000)
        manifest = {'raw_rgb_bytes':300,'tile_count':3}
        records = {'a':{'raw_rgb_bytes':100,'elapsed_seconds':10,'verified_utc':'a','output_bytes':10}}
        progress = p.Progress(budget,manifest,records)
        progress.current = 'b.tif'
        progress('downloading',50,100,force=True)
        status = json.loads((self.root/'status.json').read_text())
        self.assertEqual(status['phase_percent'],50)
        self.assertEqual(status['eta_seconds'],20)
        self.assertIsNotNone(status['estimated_completion_utc'])
        self.assertEqual(status['current_tile'],'b.tif')
        old_peak = budget.peak
        restarted = p.Budget(self.root,cap=100_000,reserve=1000)
        self.assertGreaterEqual(restarted.peak,status['peak_folder_bytes'])
        p.write_status(restarted,manifest,'paused',records)
        self.assertIsNone(json.loads((self.root/'status.json').read_text())['eta_seconds'])

    def test_rgb_geometry_and_partial_edge_masks_survive_compression(self):
        pixels = np.random.default_rng(42).integers(10, 240, (4,259,257), dtype=np.uint8)
        pixels[:, :19, :23] = 0
        transform = from_origin(600000, 4700000, 0.6, 0.6)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True, GDAL_PAM_ENABLED=False):
            with MemoryFile() as source_memory:
                with source_memory.open(driver='GTiff', width=257, height=259, count=4,
                                        dtype='uint8', crs='EPSG:26918', transform=transform, nodata=0) as writer:
                    writer.write(pixels)
                    writer.colorinterp = (ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.undefined)
                with source_memory.open() as source:
                    entry = {'name':'m_4207318_se_18_060_20220731.tif', 'url':p.BASE+'test.tif',
                             'max_output_bytes':1_000_000, 'width':257,'height':259,
                             'crs':'EPSG:26918','transform':list(transform)[:6]}
                    p.validate_source(source, entry)
                    encoded = p.encode(source, entry, 83, self.root)
                    self.assertIsNotNone(encoded)
                    try:
                        with encoded.open() as target:
                            self.assertEqual(target.count,3)
                            self.assertEqual(target.res,(0.6,0.6))
                            self.assertEqual(target.transform,transform)
                            self.assertEqual(target.crs,source.crs)
                            np.testing.assert_array_equal(target.dataset_mask(),source.dataset_mask())
                        entry['max_output_bytes'] = 1
                        self.assertIsNone(p.encode(source,entry,83,self.root))
                    finally:
                        encoded.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
