"""Older pixels fill missing coverage, with immutable base pixels on top."""
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import rasterio

from test_service import Fixtures
from imagery import Inventory,TileRenderer,lonlat_tile
from storage import DiskCache
from geospatial_search.gapfill import DATASET,REMOTE,directory
from geospatial_search.download_nys_naip import sha256


class GapDisplay(Fixtures):
    def test_saved_mixed_search_keeps_history_id_and_dataset_source_identity(self):
        from server import Application
        from shapely.geometry import box
        from geospatial_search.gapfill import BASE_DATASET

        datasets=[];results=[];rows={}
        for dataset_id in (BASE_DATASET,DATASET):
            root=self.root/dataset_id;root.mkdir()
            (root/'shards/shared').mkdir(parents=True)
            (root/'collection.json').write_text(json.dumps({'run_fingerprint':dataset_id,
                'shards':[{'shard_id':'shared','source_uri':'utm18/example.tif','chip_count':1}]}))
            prefix='' if dataset_id==BASE_DATASET else 'supplements/naip-2021-release-gapfill/'
            datasets.append(SimpleNamespace(directory=root,dataset_id=dataset_id,
                release_year=2022 if dataset_id==BASE_DATASET else 2021,
                qualify=lambda uri,prefix=prefix:prefix+uri,acquisition_date=lambda uri:'2021-09-20'))
            row={'source_uri':'utm18/example.tif','embedding_id':0,'chip_id':'shared-chip',
                'source_sha256':'synthetic','pixel_x':0,'pixel_y':0,'pixel_width':256,'pixel_height':256,
                'geometry':box(-75,43,-74.99,43.01).wkb}
            rows[dataset_id]=row
            results.append({**{key:value for key,value in row.items() if key!='geometry'},
                'dataset_id':dataset_id,'dataset_source_uri':row['source_uri'],
                'source_uri':prefix+row['source_uri'],'shard_id':'shared'})
        saved=self.root/'results.json';saved.write_text(json.dumps({'run_fingerprint':BASE_DATASET,
            'collection_fingerprints':{dataset.dataset_id:dataset.dataset_id for dataset in datasets},'results':results}))
        history_id='backend-'+'a'*32;app=Application.__new__(Application);app.job=self.root
        app.backend_history=lambda:[{'id':history_id,'backend_path':saved,'created_utc':123}]
        def selected(path,ids):return {0:rows[path.parents[2].name]}
        with patch('geospatial_search.gapfill.collections',return_value=datasets), \
             patch('geospatial_search.bundles.selected_rows',side_effect=selected), \
             patch('geospatial_search.bundles.validate_bundle',return_value={'source_uri':'utm18/example.tif','chip_count':1}):
            restored=app.saved_backend(history_id)
        self.assertEqual(restored['result_id'],history_id)
        self.assertEqual([row['dataset_id'] for row in restored['results']],[BASE_DATASET,DATASET])
        self.assertEqual(restored['results'][1]['source_uri'],'supplements/naip-2021-release-gapfill/utm18/example.tif')
        self.assertEqual(restored['results'][1]['geometry']['type'],'Polygon')

    def mixed(self,mask=False):
        base,lon,source=self.inventory(18,mask=mask)
        root=directory(base.job);(root/'imagery/utm18').mkdir(parents=True);(root/'records').mkdir()
        name='m_4307439_ne_18_060_20210920.tif';path=root/'imagery/utm18'/name
        with rasterio.open(source) as original:
            profile=original.profile;transform=original.transform
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),rasterio.open(path,'w',**profile) as ds:
            pixels=np.empty((3,1024,1024),dtype='uint8');pixels[0]=10;pixels[1]=90;pixels[2]=210
            ds.write(pixels);ds.write_mask(np.full((1024,1024),255,dtype='uint8'))
        original_record=json.loads((base.job/'records'/f'{source.name}.json').read_text())
        record={**original_record,'output':f'imagery/utm18/{name}','source_url':REMOTE+name,'output_bytes':path.stat().st_size,
            'output_sha256':sha256(path),'jpeg_quality':83,'dataset_id':DATASET,'release_year':2021,'acquisition_date':'2021-09-20'}
        (root/'records'/f'{name}.json').write_text(json.dumps(record))
        tile={'name':name,'url':REMOTE+name,'width':1024,'height':1024,'crs':'EPSG:26918','transform':list(transform)[:6]}
        manifest={'dataset_id':DATASET,'release_year':2021,'tile_count':1,'bands':[1,2,3],'resolution_m_per_pixel':.6,'tiles':[tile]}
        (root/'manifest.json').write_text(json.dumps(manifest))
        (root/'completion.json').write_text(json.dumps({'manifest_sha256':sha256(root/'manifest.json'),'tile_count':1}))
        return Inventory(base.job,strict=False),base,lon,source

    def test_full_view_preserves_base_precedence_and_base_only_fingerprint(self):
        combined,base,lon,source=self.mixed()
        self.assertEqual([s.release_year for s in combined.sources],[2021,2022])
        self.assertNotEqual(combined.fingerprint,base.fingerprint)
        base_only=Inventory(base.job,strict=False,include_supplements=False)
        self.assertEqual(base_only.fingerprint,base.fingerprint);self.assertEqual(len(base_only.sources),1)
        renderer=TileRenderer(combined,DiskCache(self.root/'cache',1_000_000))
        row=renderer.inspect(lon,43)
        self.assertEqual(row['release_year'],2022)
        image=np.asarray(Image.open(io.BytesIO(renderer.tile(*lonlat_tile(lon,43,18)))))
        np.testing.assert_allclose(image[128,128,:3],[201,57,19],atol=1)
        renderer.close()

    def test_supplement_fills_base_mask_and_native_preview_has_date_and_identity(self):
        combined,base,lon,source=self.mixed(mask=True)
        west=base.bounds[0]+.0005
        renderer=TileRenderer(combined,DiskCache(self.root/'cache',1_000_000))
        row=renderer.inspect(west,43)
        self.assertEqual(row['dataset_id'],DATASET);self.assertEqual(row['acquisition_date'],'2021-09-20')
        self.assertTrue(row['source_uri'].startswith('supplements/'))
        image=np.asarray(Image.open(io.BytesIO(renderer.preview(row))))
        np.testing.assert_equal(image[128,128,:3],[10,90,210])
        self.assertEqual(source.read_bytes()[:4],b'II*\x00')
        renderer.close()

    def test_existing_base_photographic_source_cache_is_reused(self):
        combined,base,lon,source=self.mixed()
        cache=DiskCache(self.root/'cache',1_000_000)
        base_renderer=TileRenderer(base,cache);base_renderer.photograph(base.sources[0])
        size=cache.stats()['bytes'];mixed_renderer=TileRenderer(combined,cache)
        mixed_renderer.photograph(combined.base_sources[0])
        self.assertEqual(cache.stats()['bytes'],size)
        base_renderer.close();mixed_renderer.close()
