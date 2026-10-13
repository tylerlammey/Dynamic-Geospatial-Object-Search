"""Synthetic additive-supplement provenance and shared-budget regressions."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from queue import Queue
import threading

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from geospatial_search.download_nys_naip import Budget,BudgetExceeded,encode
from geospatial_search.gapfill import ScopedBudget
from geospatial_search.download_gapfill import pilot_tiles


class GapFillTests(unittest.TestCase):
    def test_federation_merges_scores_and_keeps_dataset_qualified_rows(self):
        from geospatial_search.federated import search
        base=SimpleNamespace(dataset_id='base',directory=Path('base'))
        patch_data=SimpleNamespace(dataset_id='patch',directory=Path('patch'))
        catalog={'shards':[{'shard_id':'same','chip_count':3}],'chip_count':3}
        results={Path('base'):[('same',0,.9),('same',1,.8)],Path('patch'):[('same',0,.95),('same',1,.8)]}
        with patch('geospatial_search.search_text.search_collection',side_effect=lambda directory,*args,**kwargs:results[directory]):
            merged,details=search([(base,{},catalog),(patch_data,{},catalog)],np.array([1,0]),3)
        self.assertEqual([(dataset.dataset_id,row) for dataset,shard,row,score in merged],[('patch',0),('base',0),('base',1)])
        self.assertEqual(len(details['collections']),2)

    def test_federation_routes_exclusions_to_their_dataset_and_rejects_unknown_dataset(self):
        from geospatial_search.federated import search
        base=SimpleNamespace(dataset_id='base',directory=Path('base'))
        patch_data=SimpleNamespace(dataset_id='patch',directory=Path('patch'))
        catalog={'shards':[{'shard_id':'same','chip_count':3}],'chip_count':3}
        with patch('geospatial_search.search_text.search_collection',return_value=[]) as retrieve:
            search([(base,{},catalog),(patch_data,{},catalog)],np.array([1,0]),2,exclude_ids=[['patch','same',0]])
            self.assertEqual(retrieve.call_args_list[0].kwargs['exclude_ids'],[])
            self.assertEqual(retrieve.call_args_list[1].kwargs['exclude_ids'],[['same',0]])
            with self.assertRaisesRegex(ValueError,'dataset'):
                search([(base,{},catalog)],np.array([1,0]),2,exclude_ids=[['unknown','same',0]])

    def test_federation_rejects_different_model_normalization_or_preprocessing(self):
        from geospatial_search.federated import compatible_models
        contract={'model_id':'pinned','model_revision':'sha','arch':'arch','model_fingerprint':'weights',
            'preprocess_fingerprint':'rgb','embedding_dim':2,'metric':'cosine','embedding_dtype':'float16','amp':True}
        compatible_models(contract,dict(contract))
        for name in ('preprocess_fingerprint','metric','amp','embedding_dim'):
            with self.assertRaises(ValueError):compatible_models(contract,{**contract,name:'different'})

    def test_supplement_uses_primary_budget_and_cannot_write_base_imagery(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'supplement').mkdir()
            budget=Budget(root,cap=100,reserve=0);budget.write_bytes(root/'base.bin',b'x'*50)
            scoped=ScopedBudget(budget,root/'supplement')
            scoped.write_bytes(scoped.root/'tile.bin',b'y'*40)
            self.assertEqual(scoped.total,90)
            with self.assertRaises(BudgetExceeded):scoped.write_bytes(scoped.root/'other.bin',b'z'*20)
            with self.assertRaises(ValueError):scoped.write_bytes(root/'base.bin',b'changed')
            self.assertEqual((root/'base.bin').read_bytes(),b'x'*50)

    def test_gapfill_compression_has_correct_release_tag_and_identical_mask(self):
        with TemporaryDirectory() as folder,MemoryFile() as memory:
            transform=from_origin(500000,4800000,.6,.6)
            with memory.open(driver='GTiff',width=256,height=256,count=4,dtype='uint8',crs='EPSG:26918',transform=transform) as source:
                source.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.undefined)
                source.write(np.full((4,256,256),101,dtype=np.uint8))
                mask=np.full((256,256),255,dtype=np.uint8);mask[:20]=0;source.write_mask(mask)
                entry={'url':'https://example.invalid/tile.tif','name':'m_4307439_ne_18_060_20210920.tif',
                    'source_dataset':'NY_NAIP_2021_9664','max_output_bytes':1_000_000}
                encoded=encode(source,entry,83,Path(folder))
                try:
                    with encoded.open() as image:
                        self.assertEqual(image.tags()['SOURCE_DATASET'],'NY_NAIP_2021_9664')
                        self.assertEqual(image.tags()['SOURCE_ACQUISITION_DATE'],'20210920')
                        self.assertEqual(image.count,3);self.assertEqual(image.transform,transform)
                        np.testing.assert_array_equal(image.dataset_mask(),mask)
                finally:encoded.close()

    def test_pilot_samples_both_gap_centers_and_edges(self):
        tiles=[{'name':str(i),'nominal_center':[0,i]} for i in range(6)]
        gaps=[{'center':[0,1],'candidates':tiles[:3]},{'center':[0,4],'candidates':tiles[3:]}]
        self.assertEqual(len(pilot_tiles({'tiles':tiles,'gaps':gaps})),4)
        self.assertTrue({'1','4'}.issubset(pilot_tiles({'tiles':tiles,'gaps':gaps})))

    def test_gap_chip_selection_preserves_full_windows_and_the_validity_threshold(self):
        import torch
        from shapely.geometry import box
        from geospatial_search.embed_gapfill import prepare_source
        from geospatial_search.gapfill import DATASET
        from geospatial_search.tiling import Source,Grid
        from geospatial_search.download_nys_naip import sha256
        with TemporaryDirectory() as folder:
            path=Path(folder)/'source.tif';affine=from_origin(500000,4800000,.6,.6)
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),rasterio.open(path,'w',driver='GTiff',width=512,height=512,count=3,
                    dtype='uint8',crs='EPSG:26918',transform=affine) as ds:
                ds.write(np.full((3,512,512),100,dtype=np.uint8));ds.write_mask(np.full((512,512),255,dtype=np.uint8))
            def run():
                source=Source(path,'utm18/source.tif',512,512,'EPSG:26918',tuple(affine)[:6],sha256(path),path.stat().st_size,83)
                queue=Queue()
                prepare_source(box(500000,4799900,500110,4800000),source,Grid(),
                    {'batch_size':2,'dataset_id':DATASET,'model_id':'pinned'},lambda image:torch.zeros((3,2,2)),queue,threading.Event(),
                    'supplement-fingerprint',{'model_fingerprint':'weights','preprocess_fingerprint':'rgb'})
                values=[]
                while not queue.empty():values.append(queue.get())
                return values
            values=run();self.assertEqual(values[-1],('done',9,8))
            row=values[0][2][0];self.assertEqual((row['pixel_width'],row['pixel_height']),(256,256))
            self.assertEqual(row['dataset_id'],DATASET);self.assertEqual(row['valid_fraction'],1.0)
            with rasterio.open(path,'r+') as ds:
                mask=ds.dataset_mask();mask[:20]=0;ds.write_mask(mask)
            self.assertEqual(run(),[('done',9,9)],'Geographic eligibility must never override the 95% valid-pixel rule')


if __name__=='__main__':unittest.main()
