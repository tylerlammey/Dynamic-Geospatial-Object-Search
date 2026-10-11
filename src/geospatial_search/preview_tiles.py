"""CPU-only NYS chip previews using the production grid and RGB reader."""
from datetime import datetime, timezone
import io

from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import Window

from .download_nys_naip import Budget
from .model import load_config
from .paths import JOB
from .tiling import Footprints, Grid, load_inventory, read_rgb, validate_source

EXAMPLES = (
    ('albany','utm18/m_4207318_se_18_060_20220731.tif',6169,6110),
    ('long-island','utm18/m_4007312_ne_18_060_20220803.tif',2080,10715),
)


def save_png(budget,path,image):
    buffer=io.BytesIO()
    image.save(buffer,format='PNG')
    budget.write_bytes(path,buffer.getvalue())


def main():
    config=load_config()
    grid=Grid(config['chip_size'],config['stride'],config['min_valid_fraction'])
    sources,_=load_inventory(JOB)
    inventory={source.uri:source for source in sources}
    budget=Budget(JOB)
    destination=JOB/'tiling'/'previews'/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    metadata=[]
    with rasterio.Env(GDAL_PAM_ENABLED=False):
        for name,uri,center_x,center_y in EXAMPLES:
            source=inventory[uri]
            with rasterio.open(source.path) as dataset:
                validate_source(dataset,source)
                footprints=Footprints(dataset,grid.size)
                x=min(max(0,(center_x-grid.size//2)//grid.stride*grid.stride),source.width-grid.size-grid.stride)
                y=min(max(0,(center_y-grid.size//2)//grid.stride*grid.stride),source.height-grid.size-grid.stride)
                canvas=Image.new('RGB',(2*grid.size+12,2*grid.size+54),'black')
                for index,(dx,dy) in enumerate(((0,0),(grid.stride,0),(0,grid.stride),(grid.stride,grid.stride))):
                    window=Window(x+dx,y+dy,grid.size,grid.size)
                    image=read_rgb(dataset,window)
                    filename=f'{name}-{index+1}-native.png'
                    save_png(budget,destination/filename,image)
                    if index==0:
                        save_png(budget,destination/f'{name}-model224.png',image.resize((224,224),Image.Resampling.BICUBIC))
                    canvas.paste(image,((index%2)*(grid.size+12),42+(index//2)*(grid.size+12)))
                    spatial=footprints.metadata(window)
                    spatial['geometry_wkb_hex']=spatial.pop('geometry').hex()
                    metadata.append({'name':name,'source_uri':uri,'pixel_x':int(window.col_off),
                        'pixel_y':int(window.row_off),'size':grid.size,'stride':grid.stride,'png':filename,**spatial})
                ImageDraw.Draw(canvas).text((8,12),f'{name}: {grid.size} px, {grid.size*.6:.1f} m, overlap {100*(1-grid.stride/grid.size):.0f}%',fill='white')
                save_png(budget,destination/f'{name}-overlapping-chips.png',canvas)
    budget.atomic_json(destination/'previews.json',{'config':config,'geometry_crs':'EPSG:4326','examples':metadata})
    print(f'Previews: {destination}')
    return 0


if __name__=='__main__':
    main()
