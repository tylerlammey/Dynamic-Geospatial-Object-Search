"""Frontend request compatibility and later decision-model query controls."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch

from retrieval import validate_query,validate_search_options


class IndexedRetrievalContract(unittest.TestCase):
    def test_existing_request_keeps_its_fields_and_selects_safe_defaults(self):
        request={'query':'fighter jet','top_k':25}
        self.assertEqual(validate_query(request),('fighter jet',25,False))
        self.assertEqual(validate_search_options(request),('auto','balanced',[]))

    def test_more_matches_can_request_higher_effort_and_exclude_seen_ids(self):
        request={'query':'fighter jet','top_k':200,'search_effort':'thorough',
                 'exclude_ids':[['source-a',0],['source-b',0]]}
        self.assertEqual(validate_query(request)[1],200)
        self.assertEqual(validate_search_options(request),('auto','thorough',[['source-a',0],['source-b',0]]))

    def test_exact_mode_is_available_without_an_index(self):
        self.assertEqual(validate_search_options({'search_backend':'exact'})[0],'exact')

    def test_bad_search_controls_are_rejected(self):
        for request in ({'search_backend':'gpu'},{'search_effort':[]},{'search_effort':'unbounded'},
                        {'exclude_ids':{}},{'exclude_ids':[['a',True]]},{'exclude_ids':[['a',-1]]},
                        {'exclude_ids':[['a',0,1]]}):
            with self.assertRaises(ValueError):validate_search_options(request)

    def test_service_forwards_quality_and_seen_ids_to_existing_owned_worker(self):
        from server import Application
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'.tmp') as directory:
            app=Application.__new__(Application)
            app.frontend=Path(directory);app.search_lock=threading.Lock()
            app.search=None;app.search_log=None;app.settings={'offline_model':True}
            app.status=Mock(return_value={'busy':False,'published_chips':600,'collection_state':'complete'})
            worker=Mock();worker.poll.return_value=None
            try:
                with patch('server.subprocess.Popen',return_value=worker) as launch:
                    started=app.start_search({'query':' fighter jet ','top_k':200,
                        'search_backend':'ann','search_effort':'thorough','exclude_ids':[['source-a',7]]})
                request=json.loads((app.frontend/'logs/search'/started['task_id']/'request.json').read_text())
                self.assertEqual(request['query'],'fighter jet')
                self.assertEqual(request['top_k'],200)
                self.assertEqual(request['search_backend'],'ann')
                self.assertEqual(request['search_effort'],'thorough')
                self.assertEqual(request['exclude_ids'],[['source-a',7]])
                self.assertIn('search_worker.py',str(launch.call_args.args[0]))
                with self.assertRaisesRegex(RuntimeError,'already running'):
                    app.start_search({'query':'another query'})
            finally:
                if app.search_log:app.search_log.close()

    def test_completion_waits_for_owned_worker_exit_before_next_feedback_query(self):
        from server import Application
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'.tmp') as directory:
            app=Application.__new__(Application);app.frontend=Path(directory)
            app.search_id='a'*32;app.search=Mock();app.search.poll.return_value=None
            path=app.frontend/'logs/search'/app.search_id/'status.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'state':'complete','result_id':app.search_id}))
            self.assertEqual(app.search_progress(app.search_id)['state'],'finishing')
            app.search.poll.return_value=0
            self.assertEqual(app.search_progress(app.search_id)['state'],'complete')


if __name__=='__main__':unittest.main()
