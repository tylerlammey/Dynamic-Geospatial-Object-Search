"""Search cleanup must never touch source imagery, embeddings, or bookmarks."""
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch

from server import Application,Server
from geospatial_search.embedding_control import gpu_access


class HistoryCleanup(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=Path(__file__).parent/'.tmp')
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name).resolve()
        self.app=Application.__new__(Application)
        self.app.frontend=self.root/'frontend';self.app.frontend.mkdir()
        self.app.job=self.root/'data/nys-naip-2022-rgb-060m'
        self.app.collection=self.app.job/'embeddings/georsclip-vith14-s256-st192'
        self.app.search_lock=threading.Lock();self.app.state_lock=threading.RLock()
        self.app.search=None;self.app.search_id=None;self.app.search_log=None

    def write(self,relative,blob=b'test'):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(blob);return path

    def test_clears_history_vectors_logs_and_staging_preserving_user_data(self):
        result=self.write('frontend/state/searches/'+('a'*32)+'.json')
        vector=self.write('frontend/state/text-vectors/'+('b'*64)+'.json')
        staged=self.write('frontend/state/searches/'+('c'*32)+'.json.'+('d'*32)+'.tmp')
        logdir='frontend/logs/search/'+('e'*32)+'/'
        logs=[self.write(logdir+name) for name in ('request.json','status.json','worker.log')]
        protected=[self.write(relative) for relative in (
            'frontend/state/bookmarks.json','frontend/cache/preview.png',
            'frontend/state/searches/notes.txt','frontend/logs/desktop-service.log',
            'data/nys-naip-2022-rgb-060m/imagery/utm18/source.tif',
            'data/nys-naip-2022-rgb-060m/embeddings/shard/embeddings.f16.npy',
            'data/nys-naip-2022-rgb-060m/search-results/example/results.json')]
        self.app.search=Mock();self.app.search.poll.return_value=0
        self.app.search_log=logs[-1].open('ab')
        cleared=self.app.clear_history()
        self.assertEqual(cleared,{'removed_files':6,'freed_bytes':24})
        self.assertTrue(all(not path.exists() for path in [result,vector,staged,*logs]))
        self.assertFalse(logs[0].parent.exists())
        self.assertTrue(all(path.read_bytes()==b'test' for path in protected))
        self.assertIsNone(self.app.search_log)
        self.assertEqual(self.app.clear_history(),{'removed_files':0,'freed_bytes':0})

    def test_active_worker_blocks_cleanup_without_deleting_anything(self):
        result=self.write('frontend/state/searches/'+('a'*32)+'.json')
        self.app.search=Mock();self.app.search.poll.return_value=None
        with self.assertRaisesRegex(RuntimeError,'running search'):
            self.app.clear_history()
        self.assertTrue(result.exists())

    def test_unexpected_nested_log_directory_cancels_before_any_deletion(self):
        result=self.write('frontend/state/searches/'+('a'*32)+'.json')
        original=self.write('frontend/logs/search/'+('b'*32)+'/unexpected/source.tif')
        with self.assertRaisesRegex(ValueError,'Unexpected file'):
            self.app.clear_history()
        self.assertTrue(result.exists());self.assertTrue(original.exists())

    def test_redirected_directory_is_rejected_before_deletion(self):
        result=self.write('frontend/state/searches/'+('a'*32)+'.json')
        original_is_symlink=Path.is_symlink
        with patch.object(Path,'is_symlink',lambda path: path.name=='searches' or original_is_symlink(path)):
            with self.assertRaisesRegex(ValueError,'redirected'):
                self.app.clear_history()
        self.assertTrue(result.exists())

    def test_link_file_is_rejected_before_deletion(self):
        result=self.write('frontend/state/searches/'+('a'*32)+'.json')
        original_is_symlink=Path.is_symlink
        with patch.object(Path,'is_symlink',lambda path: path==result or original_is_symlink(path)):
            with self.assertRaisesRegex(ValueError,'regular frontend files'):
                self.app.clear_history()
        self.assertTrue(result.exists())

    def test_all_history_removes_backend_json_pngs_and_incomplete_outputs(self):
        frontend=self.write('frontend/state/searches/'+('a'*32)+'.json')
        backend='data/nys-naip-2022-rgb-060m/search-results/'
        completed=backend+'20261011T214926687552Z-airplane/'
        incomplete=backend+'20261011T215106035347Z-baseball-field/'
        artifacts=[self.write(completed+name) for name in (
            'results.json','contact-sheet.png','rank-01_score-0.2784.png','rank-02_score--0.0123.png')]
        artifacts += [self.write(incomplete+name) for name in ('results.json.new','rank-01_score-0.2456.png')]
        protected=[self.write(relative) for relative in (
            'frontend/state/bookmarks.json','frontend/cache/preview.png',
            'data/nys-naip-2022-rgb-060m/imagery/utm18/source.tif',
            'data/nys-naip-2022-rgb-060m/embeddings/shards/source/embeddings.f16.npy',
            'data/nys-naip-2022-rgb-060m/manifest.json',backend+'notes.txt')]
        self.assertEqual(self.app.clear_history(include_backend=True),{'removed_files':7,'freed_bytes':28})
        self.assertTrue(all(not path.exists() for path in [frontend,*artifacts]))
        self.assertFalse(artifacts[0].parent.exists());self.assertFalse(artifacts[-1].parent.exists())
        self.assertTrue(all(path.read_bytes()==b'test' for path in protected))
        self.assertEqual(self.app.clear_history(include_backend=True),{'removed_files':0,'freed_bytes':0})

    def test_backend_busy_blocks_all_cleanup_and_preserves_frontend_history(self):
        frontend=self.write('frontend/state/searches/'+('a'*32)+'.json')
        backend=self.write('data/nys-naip-2022-rgb-060m/search-results/example/results.json')
        with gpu_access(self.app.job):
            with self.assertRaisesRegex(RuntimeError,'using the GPU'):
                self.app.clear_history(include_backend=True)
        self.assertTrue(frontend.exists());self.assertTrue(backend.exists())

    def test_unexpected_backend_data_cancels_before_any_deletion(self):
        frontend=self.write('frontend/state/searches/'+('a'*32)+'.json')
        backend=self.write('data/nys-naip-2022-rgb-060m/search-results/example/results.json')
        original=self.write('data/nys-naip-2022-rgb-060m/search-results/example/source.tif')
        with self.assertRaisesRegex(ValueError,'Unexpected file in backend'):
            self.app.clear_history(include_backend=True)
        self.assertTrue(all(path.exists() for path in (frontend,backend,original)))

    def test_redirected_backend_directory_is_rejected_before_any_deletion(self):
        frontend=self.write('frontend/state/searches/'+('a'*32)+'.json')
        backend=self.write('data/nys-naip-2022-rgb-060m/search-results/example/results.json')
        original_is_symlink=Path.is_symlink
        with patch.object(Path,'is_symlink',lambda path: path.name=='search-results' or original_is_symlink(path)):
            with self.assertRaisesRegex(ValueError,'Backend search directory cannot be redirected'):
                self.app.clear_history(include_backend=True)
        self.assertTrue(frontend.exists());self.assertTrue(backend.exists())

    def test_resident_model_survives_all_history_cleanup_without_reacquiring_its_locks(self):
        frontend=self.write('frontend/state/searches/'+('a'*32)+'.json')
        backend=self.write('data/nys-naip-2022-rgb-060m/search-results/example/results.json')
        self.app.model_status=Mock(return_value={'state':'ready'})
        with gpu_access(self.app.job):
            self.assertEqual(self.app.clear_history(include_backend=True)['removed_files'],2)
        self.assertFalse(frontend.exists());self.assertFalse(backend.exists())
        self.assertEqual(self.app.model_status()['state'],'ready')

    def test_http_cleanup_requires_auth_confirmation_and_idle_worker(self):
        result=self.write('frontend/state/searches/'+('a'*32)+'.json')
        self.app.token='synthetic-session-token'
        server=Server(('127.0.0.1',0),self.app)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
        thread.start()
        def post(value,authenticated=True):
            connection=HTTPConnection('127.0.0.1',server.server_port,timeout=3)
            try:
                headers={'Content-Type':'application/json'}
                if authenticated:headers['Authorization']='Bearer '+self.app.token
                connection.request('POST','/api/history/clear',json.dumps(value),headers)
                response=connection.getresponse();return response.status,json.loads(response.read())
            finally:connection.close()
        try:
            confirm={'confirm':'frontend-history-only'}
            self.assertEqual(post(confirm,False)[0],403)
            self.assertEqual(post({})[0],400)
            self.app.search=Mock();self.app.search.poll.return_value=None
            self.assertEqual(post(confirm)[0],409)
            self.assertTrue(result.exists())
            self.app.search.poll.return_value=0
            self.assertEqual(post(confirm),(200,{'removed_files':1,'freed_bytes':4}))
            self.assertFalse(result.exists())
            backend=self.write('data/nys-naip-2022-rgb-060m/search-results/example/results.json')
            self.assertEqual(post({'confirm':'all-search-history'}),(200,{'removed_files':1,'freed_bytes':4}))
            self.assertFalse(backend.exists())
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':unittest.main()
