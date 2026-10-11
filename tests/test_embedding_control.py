"""Process ownership and graceful-pause acknowledgement without a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from geospatial_search.download_nys_naip import job_lock
from geospatial_search.embedding_control import gpu_access, request_pause, worker_active


class ControlTests(unittest.TestCase):
    def test_lock_rejects_another_process_and_releases_on_exit(self):
        with tempfile.TemporaryDirectory(prefix='nys-control-') as temporary:
            directory=Path(temporary)
            command=[sys.executable,'-B','-c',
                'import sys; from pathlib import Path; '
                'from geospatial_search.embedding_control import worker_active; '
                'sys.exit(0 if worker_active(Path(sys.argv[1])) else 1)',str(directory)]
            with job_lock(directory):
                self.assertEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(subprocess.run(command,capture_output=True).returncode,1)

    def test_pause_waits_for_lock_release_even_after_status_says_paused(self):
        with tempfile.TemporaryDirectory(prefix='nys-control-') as temporary:
            directory=Path(temporary)
            ready=threading.Event()
            released=threading.Event()
            def worker():
                with job_lock(directory):
                    ready.set()
                    deadline=time.monotonic()+3
                    while not (directory/'PAUSE').exists() and time.monotonic()<deadline:
                        time.sleep(.01)
                    (directory/'status.json').write_text(json.dumps({'state':'paused','durable_embedded_chips':8}))
                    time.sleep(.1)  # Simulate reader and GPU cleanup after publication.
                released.set()
            thread=threading.Thread(target=worker)
            thread.start()
            self.assertTrue(ready.wait(2))
            try:
                status=request_pause(directory,timeout=3,poll_interval=.01)
                self.assertTrue(released.wait(.1))
                self.assertFalse(worker_active(directory))
                self.assertEqual(status['durable_embedded_chips'],8)
            finally:
                thread.join(timeout=4)

    def test_pause_timeout_does_not_terminate_or_unlock_worker(self):
        with tempfile.TemporaryDirectory(prefix='nys-control-') as temporary:
            directory=Path(temporary)
            with job_lock(directory):
                with self.assertRaisesRegex(TimeoutError,'still pending'):
                    request_pause(directory,timeout=.03,poll_interval=.01)
                self.assertTrue(worker_active(directory))
                self.assertTrue((directory/'PAUSE').exists())

    def test_different_collections_cannot_use_gpu_concurrently(self):
        with tempfile.TemporaryDirectory(prefix='nys-control-') as temporary:
            directory=Path(temporary)
            with gpu_access(directory):
                with self.assertRaisesRegex(RuntimeError,'GPU'):
                    with gpu_access(directory):
                        self.fail('A competing model acquired the GPU')
            with gpu_access(directory):
                pass


if __name__=='__main__':
    unittest.main()
