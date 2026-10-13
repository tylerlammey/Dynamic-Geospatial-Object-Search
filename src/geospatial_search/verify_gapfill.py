"""Full decoded-mask coverage and labeled native-chip visual checks."""
import argparse
import io
import json

import numpy as np
from PIL import Image,ImageDraw
from pyproj import Transformer
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import Polygon,shape,mapping
from shapely.ops import transform,unary_union

from .download_nys_naip import Budget,sha256,now
from .gapfill import directory,load_inventory
from .paths import JOB


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args(argv);root=directory(JOB);budget=Budget(JOB)
    sources,_=load_inventory(JOB,require_complete=not args.pilot)
    if args.pilot:
        chosen=set(json.loads((root/'pilot.json').read_text())['tiles'])
        sources=[source for source in sources if source.path.name in chosen]
    coverage=[];checks=[];canvas=Image.new('RGB',(768,424*((len(sources)+1)//2)),'white');draw=ImageDraw.Draw(canvas)
    for index,source in enumerate(sources):
        if sha256(source.path)!=source.sha256:raise ValueError('Gap-fill source checksum differs')
        with rasterio.Env(GDAL_PAM_ENABLED=False),rasterio.open(source.path) as ds:
            if ds.tags().get('SOURCE_DATASET')!='NY_NAIP_2021_9664':raise ValueError('Gap-fill release tag differs')
            mask=ds.dataset_mask();valid=int(np.count_nonzero(mask))
            if valid==0:raise ValueError('Blank validity mask in gap-fill source')
            if valid==mask.size:
                coverage.append(Polygon([ds.transform*xy for xy in ((0,0),(ds.width,0),(ds.width,ds.height),(0,ds.height))]))
            else:
                coverage.extend(shape(geometry) for geometry,value in shapes((mask>0).astype('uint8'),mask=mask>0,transform=ds.transform) if value)
            del mask
            window=Window(max(0,source.width//2-256),max(0,source.height//2-256),512,512)
            pixels=ds.read((1,2,3),window=window).transpose(1,2,0)
            if float(pixels.std())<1:raise ValueError('Near-constant gap-fill RGB preview requires investigation')
            checks.append({'name':source.path.name,'valid_fraction':valid/(source.width*source.height),'native_resolution':ds.res,
                'source_sha256':source.sha256,'rgb_std':float(pixels.std()),'tags':ds.tags()})
            if args.pilot:
                image=Image.fromarray(pixels);image.thumbnail((384,384))
                x=(index%2)*384;y=(index//2)*424;canvas.paste(image,(x,y))
                draw.text((x+5,y+386),source.path.name,fill='black')
    manifest=json.loads((root/'manifest.json').read_text());union=unary_union(coverage)
    projector=Transformer.from_crs(4326,26918,always_xy=True)
    gaps=[]
    for gap in manifest['gaps']:
        target=transform(projector.transform,shape(gap['geometry']))
        uncovered=target.difference(union)
        gaps.append({'center':gap['center'],'area_km2':target.area/1e6,'uncovered_km2':uncovered.area/1e6,
            'valid_coverage_fraction':1-uncovered.area/target.area})
    report={'checked_utc':now(),'pilot':args.pilot,'source_count':len(sources),'sources':checks,'gaps':gaps}
    budget.atomic_json(root/'verification'/('pilot.json' if args.pilot else 'coverage.json'),report)
    if args.pilot:
        blob=io.BytesIO();canvas.save(blob,format='PNG');budget.write_bytes(root/'verification/pilot-native-chips.png',blob.getvalue())
    elif any(gap['valid_coverage_fraction']<.999 for gap in gaps):
        raise ValueError('Patch coverage is below 99.9%; inspect the saved full-mask report')
    print(json.dumps({key:value for key,value in report.items() if key!='sources'},indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
