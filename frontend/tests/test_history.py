"""Search cleanup must never touch source data, bookmarks, or CLI outputs."""
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch

from server import Application,Server


class HistoryCleanup(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=Path(__file__).parent/'.tmp')
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name).resolve()
        self.app=Application.__new__(Application)
        self.app.frontend=self.root/'frontend';self.app.frontend.mkdir()
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
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':unittest.main()
