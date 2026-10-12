"""Persistent GPU worker ownership, serialization, and one-shot compatibility."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock,patch

from server import Application
from search_worker import main,run_request
from storage import atomic_json


class ModelResidency(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=Path(__file__).parent/'.tmp')
        self.addCleanup(self.directory.cleanup)
        self.frontend=Path(self.directory.name).resolve();(self.frontend/'logs').mkdir()
        self.app=Application.__new__(Application);self.app.frontend=self.frontend
        self.app.search_lock=threading.Lock();self.app.state_lock=threading.RLock()
        self.app.search=None;self.app.search_id=None;self.app.search_log=None
        self.app.model_worker=None;self.app.model_worker_id=None;self.app.model_log=None;self.app.settings={'offline_model':True}
        self.app.status=Mock(return_value={'busy':False,'published_chips':10,'collection_state':'complete'})

    def resident(self):
        worker=Mock();worker.pid=os.getpid();worker.poll.return_value=None
        worker.stdin=io.StringIO();self.app.model_worker=worker
        self.app.model_worker_id='c'*32
        atomic_json(self.frontend/'logs/model-status.json',{'pid':worker.pid,'worker_id':self.app.model_worker_id,'state':'ready'},self.frontend)
        return worker

    def test_load_is_idempotent_and_duplicate_workers_are_not_started(self):
        worker=Mock();worker.poll.return_value=None
        try:
            with patch('server.subprocess.Popen',return_value=worker) as launch:
                self.assertEqual(self.app.start_model({}),{'state':'starting'})
                self.app.start_model({})
                self.assertEqual(launch.call_count,1)
                self.assertIn('--resident',launch.call_args.args[0])
                command=launch.call_args.args[0]
                self.assertEqual(command[command.index('--worker-id')+1],self.app.model_worker_id)
                self.assertNotIn('--online',launch.call_args.args[0])
        finally:
            if self.app.model_log:self.app.model_log.close()

    def test_resident_search_reuses_worker_blocks_overlap_and_acknowledges_without_exit(self):
        worker=self.resident()
        with patch('server.subprocess.Popen') as launch:
            started=self.app.start_search({'query':'fighter jet'})
            self.assertFalse(launch.called)
        command=json.loads(worker.stdin.getvalue())
        self.assertEqual(command['action'],'search')
        self.assertIn(started['task_id'],command['request'])
        with self.assertRaisesRegex(RuntimeError,'already running'):
            self.app.start_search({'query':'airport'})
        with self.assertRaisesRegex(RuntimeError,'running search'):self.app.stop_model()
        with self.assertRaisesRegex(RuntimeError,'running search'):self.app.clear_history()
        path=self.frontend/'logs/search'/started['task_id']/'status.json'
        atomic_json(path,{'state':'complete','result_id':started['task_id']},self.frontend)
        self.assertEqual(self.app.search_progress(started['task_id'])['state'],'complete')
        self.assertFalse(self.app.query_running())
        self.app.clear_history()
        self.assertIs(self.app.model_worker,worker,'History cleanup must preserve the loaded model')
        next_query=self.app.start_search({'query':'airport'})
        self.assertNotEqual(next_query['task_id'],started['task_id'])

    def test_stop_unloads_owned_worker_without_terminating_other_processes(self):
        worker=self.resident();worker.wait.return_value=0
        self.assertEqual(self.app.stop_model(),{'state':'unloaded'})
        self.assertIsNone(self.app.model_worker);self.assertTrue(worker.stdin.closed)
        worker.terminate.assert_not_called();worker.kill.assert_not_called()

    def test_busy_backend_and_loading_state_reject_search_and_model_loading(self):
        self.app.status.return_value['busy']=True
        with self.assertRaisesRegex(RuntimeError,'Embedding is running'):self.app.start_model({})
        with self.assertRaisesRegex(RuntimeError,'Embedding is running'):self.app.start_search({'query':'jet'})
        self.app.status.return_value['busy']=False
        worker=self.resident()
        atomic_json(self.frontend/'logs/model-status.json',{'pid':worker.pid,'worker_id':self.app.model_worker_id,'state':'loading'},self.frontend)
        with self.assertRaisesRegex(RuntimeError,'model to be ready'):self.app.start_search({'query':'jet'})
        with self.assertRaisesRegex(RuntimeError,'loading to finish'):self.app.stop_model()

    def test_stale_worker_id_cannot_claim_the_gpu_is_owned(self):
        worker=self.resident()
        atomic_json(self.frontend/'logs/model-status.json',{'pid':worker.pid,'worker_id':'d'*32,'state':'ready'},self.frontend)
        self.assertEqual(self.app.model_status(),{'state':'starting'})
        worker.poll.return_value=1
        self.assertEqual(self.app.model_status()['state'],'error')

    def test_owned_gpu_lock_is_searchable_but_stale_worker_does_not_hide_busy_backend(self):
        worker=self.resident()
        self.app.poll_lock=threading.Lock();self.app.last_status={'busy':True}
        self.app.status_at=time.monotonic();self.app.cache=Mock();self.app.cache.stats.return_value={}
        self.app.imagery_error=None;self.app.renderer=None;self.app.overview_status={}
        del self.app.status
        self.assertFalse(self.app.status()['busy'])
        atomic_json(self.frontend/'logs/model-status.json',{'pid':worker.pid,'worker_id':'d'*32,'state':'ready'},self.frontend)
        self.assertTrue(self.app.status()['busy'])

    def test_windows_launcher_pid_difference_does_not_hide_ready_model(self):
        worker=self.resident()
        atomic_json(self.frontend/'logs/model-status.json',{'pid':worker.pid+1,'worker_id':self.app.model_worker_id,'state':'ready'},self.frontend)
        self.assertEqual(self.app.model_status()['state'],'ready')

    def test_real_python_child_status_matches_launch_id_and_supports_commands(self):
        # Exercise the actual venv redirector, without allocating a GPU model.
        script=self.frontend/'fake_model_worker.py'
        script.write_text('import json,os,sys\nfrom pathlib import Path\n'
            'Path(sys.argv[1]).write_text(json.dumps({"pid":os.getpid(),"worker_id":sys.argv[2],"state":"ready"}))\n'
            'command=json.loads(sys.stdin.readline())\n'
            'assert command["action"]=="unload"\n',encoding='utf-8')
        path=self.frontend/'logs/model-status.json'
        self.app.model_worker_id='e'*32
        worker=subprocess.Popen([sys.executable,'-B',str(script),str(path),self.app.model_worker_id],
            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,encoding='utf-8',
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.app.model_worker=worker
        try:
            deadline=time.monotonic()+10
            while self.app.model_status()['state']!='ready' and time.monotonic()<deadline:time.sleep(.02)
            self.assertEqual(self.app.model_status()['state'],'ready')
            if os.name=='nt':self.assertNotEqual(self.app.model_status()['pid'],worker.pid)
            self.assertEqual(self.app.stop_model(),{'state':'unloaded'})
            self.assertEqual(worker.returncode,0)
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"action":"unload"}\n');worker.stdin.flush();worker.wait(timeout=10)
            if not worker.stdin.closed:worker.stdin.close()
            worker.stderr.close()

    def test_one_shot_and_resident_worker_publish_results_and_errors(self):
        adapter=Mock();adapter.text.return_value={'query':'jet','results':[]}
        first='a'*32;second='b'*32
        for task in (first,second):
            atomic_json(self.frontend/'logs/search'/task/'request.json',{'task_id':task,'query':'jet'},self.frontend)
        first_path=self.frontend/'logs/search'/first/'request.json'
        self.assertEqual(run_request(self.frontend,first_path,adapter),0)
        self.assertEqual(json.loads((self.frontend/'logs/search'/first/'status.json').read_text())['state'],'complete')
        adapter.resident_fingerprint='synthetic';adapter.resident_contract={'gpu':'synthetic'}
        adapter.text.side_effect=ValueError('synthetic query failure')
        commands=io.StringIO(json.dumps({'action':'search','request':f'logs/search/{second}/request.json'})+'\n'+json.dumps({'action':'unload'})+'\n')
        with patch('search_worker.ExactRetrieval',return_value=adapter),patch('sys.stdin',commands),patch('sys.argv',['worker','--frontend',str(self.frontend),'--resident','--worker-id','f'*32]):
            self.assertEqual(main(),0)
        adapter.preload.assert_called_once();adapter.close.assert_called_once()
        self.assertEqual(json.loads((self.frontend/'logs/search'/second/'status.json').read_text())['state'],'error')
        status=json.loads((self.frontend/'logs/model-status.json').read_text())
        self.assertEqual(status['state'],'unloaded');self.assertEqual(status['worker_id'],'f'*32)


if __name__=='__main__':unittest.main()
