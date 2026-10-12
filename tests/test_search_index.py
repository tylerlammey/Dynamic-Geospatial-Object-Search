"""ANN identity, exact scoring, adaptive retrieval, and integrity contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from geospatial_search import bundles
from geospatial_search.download_nys_naip import Budget
from geospatial_search.search_text import read_collection,search_collection
from test_tiling_embedding import row


@unittest.skipUnless(importlib.util.find_spec('faiss') and importlib.util.find_spec('threadpoolctl'),'Optional search dependencies are required')
class SearchIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='nys-index-test-')
        self.job=Path(self.temporary.name);self.collection=self.job/'embeddings/test'
        self.collection.mkdir(parents=True)
        rng=np.random.default_rng(43)
        self.vectors=rng.normal(size=(600,16)).astype(np.float32)
        self.vectors/=np.linalg.norm(self.vectors,axis=1,keepdims=True)
        self.vectors=self.vectors.astype(np.float16)
        budget=Budget(self.job);entries=[]
        for name,start in (('a',0),('b',300)):
            rows=[{**row(i,f'utm18/{name}.tif'),'run_fingerprint':'test'} for i in range(300)]
            run={'embedding_dim':16,'run_fingerprint':'test','source_uri':f'utm18/{name}.tif','source_sha256':'source'}
            bundles.publish_bundle(budget,self.collection/'shards'/name,self.vectors[start:start+300],rows,run)
            entries.append({'shard_id':name,'source_uri':f'utm18/{name}.tif','chip_count':300})
        root={'run_fingerprint':'test','model_contract':{'embedding_dim':16,'model_fingerprint':'model','preprocess_fingerprint':'preprocess'}}
        catalog={'run_fingerprint':'test','state':'complete','scope':'test','embedding_dim':16,
                 'chip_count':600,'shards':entries,'total_sources':2,'completed_sources':2}
        budget.atomic_json(self.collection/'run.json',root)
        budget.atomic_json(self.collection/'collection.json',catalog)
        self.catalog=catalog

    def tearDown(self):
        from geospatial_search.search_index import clear_index_cache
        clear_index_cache()
        self.temporary.cleanup()

    def build(self,threads=1):
        from geospatial_search.build_search_index import build_index
        return build_index(self.collection,self.job,partitions=2,pq_subvectors=4,training_limit=600,iterations=3,threads=threads)

    def test_all_candidates_rescore_to_existing_exact_dot_products(self):
        self.build();q=self.vectors[410].astype(np.float32);q/=np.linalg.norm(q)
        exact=search_collection(self.collection,self.catalog,q,25,backend='exact')
        indexed=search_collection(self.collection,self.catalog,q,25,backend='ann')
        self.assertEqual([(s,i) for s,i,_ in indexed],[(s,i) for s,i,_ in exact])
        np.testing.assert_allclose([x[2] for x in indexed],[x[2] for x in exact],atol=1e-6)
        self.assertEqual(indexed[0][:2],('b',110))

    def test_more_results_and_seen_ids_keep_global_identity(self):
        self.build();q=self.vectors[44].astype(np.float32);q/=np.linalg.norm(q)
        first=search_collection(self.collection,self.catalog,q,20,backend='ann')
        seen={(s,i) for s,i,_ in first}
        more=search_collection(self.collection,self.catalog,q,30,backend='ann',exclude_ids=seen,effort='thorough')
        exact=search_collection(self.collection,self.catalog,q,50,backend='exact')
        self.assertEqual([(s,i) for s,i,_ in more],[(s,i) for s,i,_ in exact if (s,i) not in seen][:30])
        self.assertFalse(seen&{(s,i) for s,i,_ in more})

    def test_fully_seen_snapshot_returns_empty_results_with_backend_diagnostics(self):
        self.build();q=self.vectors[44].astype(np.float32);q/=np.linalg.norm(q)
        seen={(s,i) for s in ('a','b') for i in range(300)};details={}
        self.assertEqual(search_collection(self.collection,self.catalog,q,10,backend='ann',exclude_ids=seen,diagnostics=details),[])
        self.assertEqual(details['backend'],'ivf-pq')
        self.assertTrue(details['exhausted'])
        self.assertEqual(details['candidate_count'],0)

    def test_shared_benchmark_references_match_independent_original_scans(self):
        from geospatial_search.benchmark_search_index import shared_exact_references
        queries=[self.vectors[i].astype(np.float32) for i in (8,398)]
        for q in queries:q/=np.linalg.norm(q)
        shared=shared_exact_references(self.collection,self.catalog,queries,50)
        for q,reference in zip(queries,shared):
            self.assertEqual(reference,search_collection(self.collection,self.catalog,q,50,backend='exact'))

    def test_candidate_rescoring_restores_existing_blas_thread_count(self):
        from threadpoolctl import threadpool_info
        self.build();q=self.vectors[13].astype(np.float32);q/=np.linalg.norm(q)
        def threads():return {x['filepath']:x['num_threads'] for x in threadpool_info() if x['user_api']=='blas'}
        before=threads()
        self.assertTrue(before)
        search_collection(self.collection,self.catalog,q,20,backend='ann')
        self.assertEqual(threads(),before)

    def test_parallel_source_readers_preserve_qualified_rows_and_scores(self):
        self.build(threads=2);q=self.vectors[339].astype(np.float32);q/=np.linalg.norm(q)
        from geospatial_search.search_index import IndexReader
        original=IndexReader._score_source;barrier=threading.Barrier(2);seen=set();lock=threading.Lock()
        def coordinated(reader,*args):
            from threadpoolctl import threadpool_info
            self.assertTrue(all(x['num_threads']==1 for x in threadpool_info() if x['user_api']=='blas'))
            with lock:seen.add(threading.get_ident())
            barrier.wait(timeout=5)
            return original(reader,*args)
        expected=search_collection(self.collection,self.catalog,q,40,backend='exact')
        with patch.object(IndexReader,'_score_source',coordinated):
            details={}
            found=search_collection(self.collection,self.catalog,q,40,backend='ann',diagnostics=details)
        self.assertEqual(len(seen),2)
        self.assertEqual(details['reader_threads'],2)
        self.assertEqual(details['rescored_count'],600)
        self.assertEqual([x[:2] for x in found],[x[:2] for x in expected])
        np.testing.assert_allclose([x[2] for x in found],[x[2] for x in expected],atol=1e-6)

    def test_sparse_mapping_and_bounded_sequential_read_score_identically(self):
        self.build();q=self.vectors[18].astype(np.float32);q/=np.linalg.norm(q)
        from geospatial_search.search_index import cached_reader
        from geospatial_search.download_nys_naip import sha256
        reader=cached_reader(str(self.collection),sha256(self.collection/'search-index/manifest.json'))
        ids=np.array([0,18,92],dtype=np.int64)
        with patch('geospatial_search.search_index.FULL_READ_FRACTION',0):
            sequential=reader._score_source(0,ids,q,3,set())
        with patch('geospatial_search.search_index.FULL_READ_FRACTION',1):
            mapped=reader._score_source(0,ids,q,3,set())
        self.assertEqual(sequential[:2],mapped[:2])
        self.assertTrue(sequential[2]);self.assertFalse(mapped[2])
        with patch('geospatial_search.search_index.MAX_SEQUENTIAL_READ_BYTES',1):
            self.assertFalse(reader._score_source(0,ids,q,3,set())[2])

    def test_missing_index_uses_exact_without_requiring_native_library(self):
        q=np.zeros(16,dtype=np.float32);q[0]=1
        diagnostics={}
        auto=search_collection(self.collection,self.catalog,q,3,diagnostics=diagnostics)
        exact=search_collection(self.collection,self.catalog,q,3,backend='exact')
        self.assertEqual(auto,exact)
        self.assertEqual(diagnostics['backend'],'exact')
        with self.assertRaisesRegex(FileNotFoundError,'index'):
            search_collection(self.collection,self.catalog,q,3,backend='ann')

    def test_changed_collection_falls_back_and_explicit_index_refuses(self):
        self.build()
        partial={**self.catalog,'chip_count':300,'shards':self.catalog['shards'][:1]}
        q=np.zeros(16,dtype=np.float32);q[0]=1
        self.assertEqual(search_collection(self.collection,partial,q,3),search_collection(self.collection,partial,q,3,backend='exact'))
        with self.assertRaisesRegex(ValueError,'snapshot'):
            search_collection(self.collection,partial,q,3,backend='ann')

    def test_index_checksum_corruption_is_rejected_before_native_load(self):
        manifest=self.build()
        path=self.collection/'search-index'/manifest['index_file']
        with path.open('ab') as handle:handle.write(b'corrupt')
        q=np.zeros(16,dtype=np.float32);q[0]=1
        with self.assertRaisesRegex(ValueError,'index'):
            search_collection(self.collection,self.catalog,q,3,backend='ann')

    def test_embedding_changes_cannot_reuse_a_saved_index(self):
        self.build()
        path=self.collection/'shards/a/embeddings.f16.npy'
        with path.open('ab') as handle:handle.write(b'changed')
        q=np.zeros(16,dtype=np.float32);q[0]=1
        with self.assertRaisesRegex(ValueError,'bundle'):
            search_collection(self.collection,self.catalog,q,3,backend='ann')

    def test_invalid_query_and_unknown_seen_id_fail(self):
        self.build();q=np.zeros(16,dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'normalized'):
            search_collection(self.collection,self.catalog,q,3,backend='ann')
        q[0]=1
        with self.assertRaises(ValueError):
            search_collection(self.collection,self.catalog,q,3,backend='ann',exclude_ids={('missing',0)})

    def test_effort_expands_candidate_pool_and_cached_reader_rechecks_bundles(self):
        self.build();q=self.vectors[13].astype(np.float32);q/=np.linalg.norm(q)
        from geospatial_search.search_index import cached_reader
        counts=[]
        with patch('geospatial_search.search_index.CANDIDATE_FLOORS',{'fast':128,'balanced':256,'thorough':512}):
            for effort in ('fast','balanced','thorough'):
                details={}
                search_collection(self.collection,self.catalog,q,3,backend='ann',effort=effort,diagnostics=details)
                counts.append(details['candidate_count'])
        self.assertEqual(counts,[128,256,512])
        self.assertEqual(cached_reader.cache_info().currsize,1)
        path=self.collection/'shards/a/chips.geoparquet'
        with path.open('ab') as handle:handle.write(b'changed after first search')
        with self.assertRaisesRegex(ValueError,'bundle'):
            search_collection(self.collection,self.catalog,q,3,backend='ann')

    def test_native_dependency_failure_falls_back_without_hiding_corruption(self):
        self.build();q=np.zeros(16,dtype=np.float32);q[0]=1
        from geospatial_search.search_index import clear_index_cache,IndexDependencyUnavailable
        clear_index_cache()
        with patch('geospatial_search.search_index.IndexReader',side_effect=IndexDependencyUnavailable('Native dependency unavailable')):
            self.assertEqual(search_collection(self.collection,self.catalog,q,3),search_collection(self.collection,self.catalog,q,3,backend='exact'))
            with self.assertRaises(IndexDependencyUnavailable):search_collection(self.collection,self.catalog,q,3,backend='ann')


if __name__=='__main__':unittest.main()
