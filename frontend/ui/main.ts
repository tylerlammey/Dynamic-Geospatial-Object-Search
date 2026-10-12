import L from 'leaflet';
import type {Polygon} from 'geojson';
import './style.css';

type Chip={rank:number|null;score:number|null;shard_id:string;embedding_id:number|null;chip_id:string;source_uri:string;source_sha256:string;pixel_x:number;pixel_y:number;pixel_width:number;pixel_height:number;centroid_lon:number;centroid_lat:number;west:number;south:number;east:number;north:number;geometry:Polygon;crs_wkt:string};
type ResultSet={query:string;result_id?:string;run_fingerprint:string;results:Chip[];searched_source_tiles:number;total_source_tiles:number;searched_chip_count:number;searched_source_uris?:string[];partial_collection:boolean;created_utc:number;search?:{backend:string;approximate:boolean;search_seconds:number}};
type Status={state:string;collection_state:string;busy:boolean;published_sources:number;published_chips:number;total_sources:number;coverage:string[];run_fingerprint?:string;partial:boolean;error?:string;imagery_error?:string;model?:{state:string;error?:string;gpu?:string};cache:{bytes:number;limit_bytes:number;files:number};elapsed_hms?:string;eta_hms?:string;search_index?:{ready:boolean;algorithm:string};overview?:{state:string;ready:boolean;epoch:number;completed_sources?:number;total_sources?:number;eta_seconds?:number;error?:string}};
type Configuration={imagery:{bounds:[number,number,number,number];count:number}|null;imagery_error?:string;sources:{uri:string;bounds:[number,number,number,number]}[];attribution:string};
const $=<T extends HTMLElement=HTMLElement>(selector:string)=>{const value=document.querySelector<T>(selector);if(!value)throw new Error(`Missing UI control: ${selector}`);return value;};
const escape=(value:unknown)=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const fragment=location.hash.slice(1);
if(fragment)sessionStorage.setItem('session-token',fragment);
const token=fragment||sessionStorage.getItem('session-token')||'';
history.replaceState(null,'',location.pathname+location.search);
const url=(route:string)=>`${route}${route.includes('?')?'&':'?'}token=${encodeURIComponent(token)}`;
async function api<T>(route:string,value?:unknown):Promise<T>{
  const response=await fetch(route,{method:value===undefined?'GET':'POST',headers:{Authorization:`Bearer ${token}`,...(value===undefined?{}:{'Content-Type':'application/json'})},body:value===undefined?undefined:JSON.stringify(value)});
  const data=await response.json();if(!response.ok)throw new Error(data.error||`Local service returned ${response.status}`);return data as T;
}
const state:{results:Chip[];session:ResultSet|null;selected:Chip|null;page:number;coverage:boolean;status:Status|null;config:Configuration|null;bookmarks:Chip[];task:string|null;allowPartial:boolean}={results:[],session:null,selected:null,page:0,coverage:false,status:null,config:null,bookmarks:[],task:null,allowPartial:false};
const PAGE=20;
let modelAction=false;
let toastTimer:number|undefined;
function toast(message:string){$('#toast').textContent=message;$('#toast').classList.remove('hidden');window.clearTimeout(toastTimer);toastTimer=window.setTimeout(()=>$('#toast').classList.add('hidden'),5500);}
function notice(message:string){$('#notice').textContent=message;$('#notice').classList.toggle('hidden',!message);}
function run(action:()=>Promise<unknown>){void action().catch(error=>{notice(error.message);toast(error.message);});}
const map=L.map('map',{zoomControl:false,minZoom:5,maxZoom:20,preferCanvas:true,zoomSnap:1,zoomAnimation:true});
L.control.zoom({position:'topright'}).addTo(map);L.control.scale({imperial:false,position:'bottomleft',maxWidth:100}).addTo(map);
map.attributionControl.setPrefix(false);map.attributionControl.addAttribution('Courtesy USDA-FSA; distributed by NOAA Digital Coast');
map.setView([42.7,-75.6],7);
const overlays=L.layerGroup().addTo(map);
const coverageLayer=L.layerGroup();
const selectedLayer=L.layerGroup().addTo(map);
const footprints=L.layerGroup().addTo(map);

type TileJob={img:HTMLImageElement;coords:L.Coords;done:L.DoneCallback;controller:AbortController;cancelled:boolean;objectUrl?:string};
class TilePool{
  queue:TileJob[]=[];active=0;failures=0;loaded=0;
  enqueue(job:TileJob){this.queue.push(job);this.pump();}
  pump(){
    this.queue=this.queue.filter(job=>!job.cancelled);
    while(this.active<4&&this.queue.length){const job=this.queue.shift()!;this.active++;void this.fetchTile(job).finally(()=>{this.active--;this.pump();});}
    $('#tile-status').textContent=this.active+this.queue.length?`Loading local imagery · ${this.active} active${this.queue.length?` / ${this.queue.length} waiting`:''}`:this.failures?'Some tiles could not load. Pan or use Fit state to retry.':'';
  }
  async fetchTile(job:TileJob){
    try{
      if(job.cancelled)return;
      const revision=`${state.status?.overview?.epoch||0}-${state.status?.overview?.ready?'ready':'preparing'}`;
      const route=url(`/api/tiles/${job.coords.z}/${job.coords.x}/${job.coords.y}.png?display=${revision}`);
      let response=await fetch(route,{signal:job.controller.signal});
      for(let retry=0;response.status===503&&retry<2&&!job.cancelled;retry++){
        await new Promise(resolve=>setTimeout(resolve,500*(retry+1)));response=await fetch(route,{signal:job.controller.signal});
      }
      if(!response.ok){const error=await response.json();throw new Error(error.error||'Tile read failed');}
      const blob=await response.blob();if(job.cancelled)return;
      job.objectUrl=URL.createObjectURL(blob);
      job.img.onload=()=>{if(!job.cancelled){this.loaded++;job.done(undefined,job.img);}};
      job.img.onerror=()=>{if(!job.cancelled)job.done(new Error('Tile image could not decode'),job.img);};
      job.img.src=job.objectUrl;
    }catch(error){if(!job.cancelled){this.failures++;job.done(error as Error,job.img);}}
  }
}
const tilePool=new TilePool();
const jobs=new WeakMap<HTMLElement,TileJob>();
const LocalTiles=L.GridLayer.extend({
  createTile(coords:L.Coords,done:L.DoneCallback){const img=document.createElement('img');img.alt='';img.setAttribute('role','presentation');const job={img,coords,done,controller:new AbortController(),cancelled:false};jobs.set(img,job);tilePool.enqueue(job);return img;},
  _abortLoading(this:L.GridLayer & {_tiles:Record<string,{coords:L.Coords}>;_tileZoom:number;_removeTile:(key:string)=>void}){
    // GridLayer retains unloaded old levels for its fade. Abort them as soon
    // as zoom changes, rather than letting them starve the current viewport.
    for(const [key,tile] of Object.entries(this._tiles||{}))if(tile.coords.z!==this._tileZoom)this._removeTile(key);
  }
});
const TileClass=LocalTiles as typeof L.GridLayer;
const tiles:L.GridLayer=new TileClass({tileSize:256,keepBuffer:1,updateWhenIdle:true,updateWhenZooming:false,bounds:[[-1,-180],[85,180]],minZoom:5,maxZoom:20});
tiles.on('tileunload',(event:L.TileEvent)=>{const job=jobs.get(event.tile);if(job){job.cancelled=true;job.controller.abort();if(job.objectUrl)URL.revokeObjectURL(job.objectUrl);tilePool.pump();}});

function geographicBounds(bounds:[number,number,number,number]):L.LatLngBounds{return L.latLngBounds([bounds[1],bounds[0]],[bounds[3],bounds[2]]);}
function fitState(){const bounds=state.config?.imagery?.bounds;if(bounds)map.fitBounds(geographicBounds(bounds),{padding:[25,40],maxZoom:8});}
function chipBounds(row:Chip){return geographicBounds([row.west,row.south,row.east,row.north]);}
function chipKey(row:Chip){return row.embedding_id===null?row.chip_id:`${row.shard_id}:${row.embedding_id}`;}
function previewUrl(row:Chip,context=false){return url(`/api/preview?source=${encodeURIComponent(row.source_uri)}&x=${row.pixel_x}&y=${row.pixel_y}&context=${context?1:0}`);}
function detailLevel(){
  const zoom=map.getZoom();
  $('#detail-level').textContent=zoom<=10?'Statewide aerial overview':zoom<14?'Aerial overview':zoom<18?'Downloaded aerial imagery':'Native-detail aerial imagery';
  const meters=156543.03392*Math.cos(map.getCenter().lat*Math.PI/180)/2**zoom;
  $('#coordinates').textContent=`Lon ${map.getCenter().lng.toFixed(5)} / Lat ${map.getCenter().lat.toFixed(5)}`;
  $('#display-resolution').textContent=`Display ≈ ${meters.toFixed(meters<1?2:1)} m/px`;
}
function showCoverage(){
  coverageLayer.clearLayers();
  if(!state.coverage){map.removeLayer(coverageLayer);return;}
  const uris=new Set(state.session?.searched_source_uris||state.status?.coverage||[]);
  for(const source of state.config?.sources||[])if(uris.has(source.uri))L.rectangle(geographicBounds(source.bounds),{color:'#e9ce75',weight:.5,fillColor:'#edcf68',fillOpacity:.13,interactive:false}).addTo(coverageLayer);
  coverageLayer.addTo(map);
  $('#coverage-toggle').textContent=state.session?.searched_source_uris?'Hide searched snapshot coverage':'Hide published search coverage';
}
function renderOverlays(){
  overlays.clearLayers();footprints.clearLayers();selectedLayer.clearLayers();
  const selected=state.selected;
  if(selected)L.geoJSON(selected.geometry,{style:{color:'#ffc951',weight:3,fillColor:'#ffc951',fillOpacity:.08}}).addTo(selectedLayer);
  const visible=state.results.filter(row=>map.getBounds().pad(.15).contains([row.centroid_lat,row.centroid_lon]));
  if(map.getZoom()>=14)for(const row of visible)L.geoJSON(row.geometry,{style:{color:chipKey(row)=== (selected&&chipKey(selected))?'#ffc951':'#f6faf4',weight:1,fillOpacity:.02},onEachFeature:(_,layer)=>layer.on('click',event=>{L.DomEvent.stopPropagation(event);select(row);})}).addTo(footprints);
  const grouped=new Map<string,Chip[]>();
  for(const row of visible){const pixel=map.latLngToContainerPoint([row.centroid_lat,row.centroid_lon]);const cell=map.getZoom()>=16?40:65;const key=`${Math.floor(pixel.x/cell)}:${Math.floor(pixel.y/cell)}`;const group=grouped.get(key)||[];group.push(row);grouped.set(key,group);}
  for(const rows of grouped.values()){
    const lon=rows.reduce((s,r)=>s+r.centroid_lon,0)/rows.length;const lat=rows.reduce((s,r)=>s+r.centroid_lat,0)/rows.length;
    const chosen=!!selected&&rows.some(r=>chipKey(r)===chipKey(selected));
    const marker=L.marker([lat,lon],{icon:L.divIcon({className:`rank-marker${rows.length>1?' cluster':''}${chosen?' selected':''}`,html:String(rows.length>1?rows.length:rows[0].rank),iconSize:rows.length>1?[36,36]:[29,29]}),keyboard:true,title:rows.length>1?`${rows.length} nearby matches`:`Rank ${rows[0].rank}`}).addTo(overlays);
    marker.on('click',()=>{
      if(rows.length===1){select(rows[0]);return;}
      const popup=document.createElement('div');const title=document.createElement('strong');title.textContent=`${rows.length} nearby matches`;popup.append(title);
      const caption=document.createElement('p');caption.textContent='Every overlapping chip is retained.';popup.append(caption);
      const list=document.createElement('div');list.className='cluster-list';popup.append(list);
      for(const row of rows){const button=document.createElement('button');button.textContent=`#${row.rank}`;button.addEventListener('click',()=>{map.closePopup();select(row);});list.append(button);}
      const fit=document.createElement('button');fit.textContent='Zoom to these matches';fit.addEventListener('click',()=>{map.closePopup();map.fitBounds(L.latLngBounds(rows.map(r=>[r.centroid_lat,r.centroid_lon] as L.LatLngTuple)),{padding:[50,70],maxZoom:17});});popup.append(fit);
      marker.bindPopup(popup).openPopup();
    });
  }
}
function renderResults(){
  $('#results-title').textContent=state.session?`“${state.session.query}”`:'Search matches';$('#result-count').textContent=String(state.results.length);
  if(!state.session){
    $('#result-list').innerHTML='<div class="empty-state"><strong>Your imagery is ready to explore.</strong><p>Run a text search or click downloaded imagery to inspect a native chip.</p></div>';
    $('.pagination').classList.add('hidden');$('#page-label').textContent='';return;
  }
  const start=state.page*PAGE;const page=state.results.slice(start,start+PAGE);
  $('#result-list').innerHTML=page.length?page.map(row=>`<button class="result-card${state.selected&&chipKey(row)===chipKey(state.selected)?' selected':''}" data-rank="${row.rank}" aria-pressed="${!!state.selected&&chipKey(row)===chipKey(state.selected)}"><img loading="lazy" src="${escape(previewUrl(row))}" alt="Native aerial chip, rank ${row.rank}"><span class="info"><span class="rank">${String(row.rank).padStart(2,'0')} · ${row.centroid_lat.toFixed(3)}° N</span><span class="score">${row.score!.toFixed(4)} similarity</span><span class="coords">${row.centroid_lon.toFixed(4)}, ${row.centroid_lat.toFixed(4)}</span></span></button>`).join(''):'<div class="empty-state"><strong>No matches.</strong><p>Try another phrase, and check the searched coverage.</p></div>';
  $('#result-list').querySelectorAll<HTMLButtonElement>('[data-rank]').forEach(button=>button.addEventListener('click',()=>select(state.results.find(row=>row.rank===Number(button.dataset.rank))!)));
  $('.pagination').classList.toggle('hidden',state.results.length<=PAGE);$('#page-label').textContent=`${start+1}–${Math.min(start+PAGE,state.results.length)} of ${state.results.length}`;
  $<HTMLButtonElement>('#page-prev').disabled=state.page===0;$<HTMLButtonElement>('#page-next').disabled=start+PAGE>=state.results.length;
}
function renderSelection(){
  const row=state.selected;if(!row){
    $('#selection-kind').textContent='NATIVE WINDOW';
    $('#selection').innerHTML='<div class="empty-state">Click the map or select a search result to see the actual native-resolution chip here.</div>';return;
  }
  const isSearch=row.rank!==null;const bookmarked=state.bookmarks.some(b=>chipKey(b)===chipKey(row));
  $('#selection-kind').textContent=isSearch?'SEARCH MATCH':'MAP LOCATION';
  $('#selection').innerHTML=`<div class="selection-nav"><span>${isSearch?`Rank ${row.rank} of ${state.results.length}`:'Selected map location'}</span><button id="previous" aria-label="Previous result" ${isSearch?'':'disabled'}>←</button><button id="next" aria-label="Next result" ${isSearch?'':'disabled'}>→</button></div><div class="inspector-main"><div><div class="preview-label"><span>256 × 256 source pixels</span><span>Actual imagery</span></div><img class="selected-preview" src="${escape(previewUrl(row))}" alt="Selected actual NAIP chip"><div class="context-wrap"><img id="context-preview" src="${escape(previewUrl(row,true))}" alt="Surrounding imagery, gold chip outline" tabindex="0" role="button"><span>Surrounding imagery<br><span class="muted">Gold outline = selected chip</span></span></div></div><div><h3>${isSearch?`Match ${row.rank}`:'Map location'}</h3>${isSearch?`<div class="selection-score">${row.score!.toFixed(4)} <span class="small">SIMILARITY</span></div><p class="score-note">A ranking score, not detection confidence.</p>`:'<p class="score-note">Native chip from your downloaded imagery.</p>'}<dl class="details"><div><dt>Source</dt><dd>2022 NAIP · RGB</dd></div><div><dt>Native resolution</dt><dd>0.6 m / pixel</dd></div><div><dt>Chip width</dt><dd>153.6 m</dd></div><div><dt>Longitude</dt><dd>${row.centroid_lon.toFixed(6)}</dd></div><div><dt>Latitude</dt><dd>${row.centroid_lat.toFixed(6)}</dd></div><div><dt>Pixel window</dt><dd>${row.pixel_x}, ${row.pixel_y}<br>256 × 256</dd></div></dl><button class="bookmark" id="bookmark">${bookmarked?'★ Remove bookmark':'☆ Bookmark this chip'}</button></div></div><p class="inspector-foot">${escape(row.source_uri)}<br>${isSearch?`Shard ${escape(row.shard_id)} / row ${row.embedding_id}`:'Source grid preserved. Display map uses Web Mercator.'}</p>`;
  const cycle=(delta:number)=>{if(!isSearch)return;const index=state.results.findIndex(r=>chipKey(r)===chipKey(row));select(state.results[(index+delta+state.results.length)%state.results.length]);};
  $('#previous').addEventListener('click',()=>cycle(-1));$('#next').addEventListener('click',()=>cycle(1));
  $('#bookmark').addEventListener('click',()=>run(async()=>{state.bookmarks=await api<Chip[]>('/api/bookmarks',row);renderSelection();}));
  const context=()=>openDialog('Surrounding actual imagery',`<img class="dialog-preview" src="${escape(previewUrl(row,true))}" alt="Selected chip and nearby imagery"><p>Gold outline: the original 256 × 256 pixel chip. This view shows surrounding native imagery; it does not change the embedding window.</p>`);
  $('#context-preview').addEventListener('click',context);$('#context-preview').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();context();}});
  $('#selection').querySelectorAll<HTMLImageElement>('img').forEach(img=>img.addEventListener('error',()=>toast('The chip preview could not load. Check the imagery folder or local service.')));
}
function select(row:Chip,zoom=true){
  state.selected=row;if(row.rank!==null)state.page=Math.floor((row.rank-1)/PAGE);renderResults();renderSelection();renderOverlays();
  if(zoom)map.fitBounds(chipBounds(row).pad(.9),{maxZoom:18,padding:[40,75]});
}
async function loadResults(identifier:string){
  const results=await api<ResultSet>(`/api/results?id=${encodeURIComponent(identifier)}`);state.session=results;state.results=results.results;state.page=0;
  $<HTMLInputElement>('#query').value=results.query;
  $('#search-scope').textContent=`Results searched ${results.searched_source_tiles.toLocaleString()} / ${results.total_source_tiles.toLocaleString()} source tiles${results.partial_collection?' · partial snapshot':''}`;
  if(results.search)$('#search-scope').textContent+=` · ${results.search.approximate?'Indexed':'Exact'} retrieval ${results.search.search_seconds.toFixed(2)}s`;
  renderResults();renderOverlays();showCoverage();if(state.results.length)select(state.results[0],false);
}
function renderStatus(){
  const status=state.status;if(!status)return;
  const model=status.model?.state||'unloaded';const loading=['starting','loading'].includes(model);const resident=['ready','searching'].includes(model);
  $<HTMLButtonElement>('#load-model').textContent=loading?'Loading model…':resident?'Model on GPU':'Load model to GPU';
  $<HTMLButtonElement>('#load-model').disabled=modelAction||loading||resident||model==='unloading'||!!state.task||status.busy||!status.published_chips;
  $('#unload-model').classList.toggle('hidden',!resident&&model!=='unloading');
  $<HTMLButtonElement>('#unload-model').disabled=modelAction||!!state.task||model==='searching'||model==='unloading';
  $('#coverage-text').textContent=`${status.published_sources.toLocaleString()} of ${status.total_sources.toLocaleString()} source tiles searchable · ${status.published_chips.toLocaleString()} saved chips`;
  $('#coverage-pill').textContent=status.partial?'Partial coverage':'Complete coverage';$('#status-dot').style.background=status.busy?'#b48736':'#4d9668';
  $('#footer-worker').textContent=status.busy?`Embedding / GPU busy · imagery available${status.elapsed_hms?` · elapsed ${status.elapsed_hms}`:''}`:`${status.collection_state||status.state} collection · ${status.search_index?.ready?'fast indexed':'exact'} search available`;
  if(loading)$('#footer-worker').textContent='Loading GeoRSCLIP onto the GPU · imagery browsing remains available';
  else if(resident)$('#footer-worker').textContent='GeoRSCLIP stays on GPU · unload the model before embedding or CLI search';
  $('.search-meta').textContent=status.search_index?.ready?'GeoRSCLIP · fast search / exact candidate scores':'GeoRSCLIP · exact similarity search';
  $('#cache-status').textContent=`Cache ${(status.cache.bytes/1e9).toFixed(2)} / ${(status.cache.limit_bytes/1e9).toFixed(2)} GB`;
  const overview=status.overview;
  if(overview){
    const building=['starting','preparing','publishing'].includes(overview.state);
    $('#overview-status').classList.toggle('hidden',overview.ready&&!building);
    $('#overview-status').textContent=building?`Preparing photographic overview · ${overview.completed_sources||0} / ${overview.total_sources||3805} source tiles${overview.eta_seconds?` · about ${Math.ceil(overview.eta_seconds/60)} min remaining`:''}`:overview.error?`Overview: ${overview.error}`:'Prepare the photographic overview in Settings for statewide viewing. Close-up imagery is available now.';
  }
  $<HTMLButtonElement>('#search-button').disabled=!!state.task||!status.published_chips||modelAction||loading||model==='unloading';
  if(!state.task){
    if(status.imagery_error)notice(`Local imagery unavailable: ${status.imagery_error}`);
    else if(status.error)notice(status.error);
    else if(status.model?.error)notice(`GPU model: ${status.model.error}`);
    else if(loading)notice('Loading and warming up the pinned GeoRSCLIP model once. Later searches will reuse it until you unload it or close the app.');
    else if(status.busy){$('#notice').innerHTML='Embedding is running; pause it using the existing command before searching. Run <code>.\\scripts\\embed-nys.ps1 -Pause</code> yourself and wait for acknowledgement. Imagery browsing and saved results remain available.';$('#notice').classList.remove('hidden');}
    else if(!status.published_chips)notice('No saved embeddings yet. Browse the imagery while the pipeline publishes its first complete source bundle.');
    else notice('');
  }
}
async function refreshStatus(){const before=state.status?.overview;state.status=await api<Status>('/api/status');renderStatus();if(state.coverage&&!state.session?.searched_source_uris)showCoverage();const after=state.status.overview;if(before&&after&&(before.ready!==after.ready||before.epoch!==after.epoch)){tiles.redraw();if(after.ready)toast('Photographic overview is ready. Statewide and county views now use cached aerial imagery.');}}
async function pollModel(){
  await refreshStatus();
  if(['starting','loading'].includes(state.status?.model?.state||''))window.setTimeout(()=>run(pollModel),1000);
  else if(state.status?.model?.state==='ready')toast('GeoRSCLIP is ready on the GPU. Searches will reuse this model.');
}
$('#load-model').addEventListener('click',()=>run(async()=>{
  if(modelAction)return;modelAction=true;renderStatus();
  try{await api('/api/model/load',{allow_partial:state.allowPartial});await pollModel();}
  finally{modelAction=false;renderStatus();}
}));
$('#unload-model').addEventListener('click',()=>run(async()=>{
  if(modelAction)return;modelAction=true;renderStatus();
  try{await api('/api/model/unload',{});await refreshStatus();toast('Model unloaded; the GPU is available again.');}
  finally{modelAction=false;renderStatus();}
}));
async function pollSearch(identifier:string){
  const progress=await api<{state:string;error?:string;result_id?:string;scanned_sources?:number;searched_source_tiles?:number;total_source_tiles?:number}>(`/api/search-progress?id=${identifier}`);
  if(progress.state==='complete'){state.task=null;$('#progress').classList.add('hidden');$<HTMLButtonElement>('#search-button').textContent='Search imagery';await loadResults(progress.result_id!);await refreshStatus();toast(`Search finished: ${state.results.length} results from a fixed published snapshot.`);return;}
  if(progress.state==='error'){state.task=null;$('#progress').classList.add('hidden');$<HTMLButtonElement>('#search-button').textContent='Search imagery';renderStatus();throw new Error(progress.error);}
  const scanned=progress.scanned_sources||0;const total=progress.searched_source_tiles||state.status?.published_sources||1;
  notice(progress.state==='encoding'?'Encoding text with the pinned GeoRSCLIP model…':progress.state==='loading_index'?'Loading the local search index…':progress.state==='finishing'?'Finalizing results…':`Searching published coverage: ${scanned.toLocaleString()} of ${total.toLocaleString()} source tiles. You can keep exploring the map.`);
  $('#progress i').style.width=`${Math.max(2,scanned/total*100)}%`;
  window.setTimeout(()=>run(()=>pollSearch(identifier)),750);
}
$('#search-form').addEventListener('submit',event=>{event.preventDefault();run(async()=>{
  if(state.task)return;
  const mode=$<HTMLSelectElement>('#search-mode').value;
  const result=await api<{task_id:string}>('/api/search',{query:$<HTMLInputElement>('#query').value,top_k:Number($<HTMLInputElement>('#top-k').value),allow_partial:state.allowPartial,search_backend:mode==='exact'?'exact':'auto',search_effort:mode==='thorough'?'thorough':'balanced'});
  state.task=result.task_id;renderStatus();$('#progress').classList.remove('hidden');$<HTMLButtonElement>('#search-button').textContent='Searching…';await pollSearch(result.task_id);
});});
$('#home').addEventListener('click',fitState);$('#zoom-selection').addEventListener('click',()=>state.selected?select(state.selected):toast('Click downloaded imagery to select a chip.'));
$('#fit-matches').addEventListener('click',()=>state.results.length?map.fitBounds(L.latLngBounds(state.results.map(r=>[r.centroid_lat,r.centroid_lon] as L.LatLngTuple)),{padding:[40,75],maxZoom:17}):toast('Run a search or open saved results first.'));
$('#coverage-toggle').addEventListener('click',()=>{state.coverage=!state.coverage;$('#coverage-toggle').setAttribute('aria-pressed',String(state.coverage));if(!state.coverage)$('#coverage-toggle').textContent='Show search coverage';showCoverage();});
$('#page-prev').addEventListener('click',()=>{state.page=Math.max(0,state.page-1);renderResults();});$('#page-next').addEventListener('click',()=>{state.page++;renderResults();});
map.on('moveend zoomend',()=>{renderOverlays();detailLevel();});
map.on('mousemove',(event:L.LeafletMouseEvent)=>{$('#coordinates').textContent=`Lon ${event.latlng.lng.toFixed(5)} / Lat ${event.latlng.lat.toFixed(5)}`;});
let inspectController:AbortController|null=null;
map.on('click',(event:L.LeafletMouseEvent)=>{run(async()=>{
  inspectController?.abort();inspectController=new AbortController();
  const response=await fetch(`/api/inspect?lon=${event.latlng.lng}&lat=${event.latlng.lat}`,{headers:{Authorization:`Bearer ${token}`},signal:inspectController.signal});
  const row=await response.json();if(!response.ok)throw new Error(row.error);select(row,false);
});});
function openDialog(title:string,content:string){$('#dialog-title').textContent=title;$('#dialog-content').innerHTML=content;$<HTMLDialogElement>('#dialog').showModal();}
$('#close-dialog').addEventListener('click',()=>$<HTMLDialogElement>('#dialog').close());
async function showHistory(){
  const records=await api<{id:string;query:string;count:number;searched_source_tiles:number;total_source_tiles:number;created_utc:number}[]>('/api/history');
  const list=records.length?`<div class="dialog-list">${records.map(r=>`<button data-history="${r.id}">${escape(r.query)}<small>${r.count} matches · searched ${r.searched_source_tiles} / ${r.total_source_tiles} source tiles · ${new Date(r.created_utc*1000).toLocaleString()}</small></button>`).join('')}</div>`:'<p>No saved searches yet. Once a query finishes, you can reopen it here while embedding runs.</p>';
  openDialog('Saved searches',`${list}<p><button id="clear-history" ${state.task?'disabled':''}>Clear all search history…</button></p><p class="muted">Removes frontend and command-line saved searches, generated search previews, and the currently displayed matches.</p>`);
  $('#dialog-content').querySelectorAll<HTMLButtonElement>('[data-history]').forEach(button=>button.addEventListener('click',()=>run(async()=>{
    $<HTMLDialogElement>('#dialog').close();await loadResults(button.dataset.history!);
  })));
  $('#clear-history').addEventListener('click',()=>{
    openDialog('Clear all search history?', '<p>This removes frontend and backend saved searches, generated backend search previews/contact sheets, cached query vectors, and search logs to free storage. It also clears the current results and map markers. Bookmarks, source imagery, embeddings, and the loaded GPU model are preserved.</p><p><button id="confirm-history" class="primary">Clear all search history</button></p>');
    $('#confirm-history').addEventListener('click',()=>run(async()=>{
      $<HTMLButtonElement>('#confirm-history').disabled=true;
      try{
        const cleared=await api<{removed_files:number;freed_bytes:number}>('/api/history/clear',{confirm:'all-search-history'});
        state.results=[];state.session=null;state.selected=null;state.page=0;state.coverage=false;
        $<HTMLInputElement>('#query').value='';$('#search-scope').textContent='';map.closePopup();
        $('#coverage-toggle').setAttribute('aria-pressed','false');$('#coverage-toggle').textContent='Show search coverage';
        $('#progress').classList.add('hidden');
        renderResults();renderSelection();renderOverlays();showCoverage();renderStatus();
        await showHistory();toast(`Search history cleared: ${(cleared.freed_bytes/1e6).toFixed(1)} MB freed.`);
      }catch(error){if(document.querySelector('#confirm-history'))$<HTMLButtonElement>('#confirm-history').disabled=false;throw error;}
    }));
  });
}
$('#history').addEventListener('click',()=>run(showHistory));
$('#bookmarks').addEventListener('click',()=>run(async()=>{state.bookmarks=await api<Chip[]>('/api/bookmarks');openDialog('Bookmarked chips',state.bookmarks.length?`<div class="dialog-list">${state.bookmarks.map((r,i)=>`<button data-bookmark="${i}">${r.centroid_lon.toFixed(5)}, ${r.centroid_lat.toFixed(5)}<small>${escape(r.source_uri)}</small></button>`).join('')}</div>`:'<p>Select a native chip, then choose “Bookmark this chip.” Bookmarks are saved locally under frontend/state/.</p>');$('#dialog-content').querySelectorAll<HTMLButtonElement>('[data-bookmark]').forEach(button=>button.addEventListener('click',()=>{$<HTMLDialogElement>('#dialog').close();select(state.bookmarks[Number(button.dataset.bookmark)]);}));}));
$('#settings').addEventListener('click',()=>{openDialog('Local imagery & cache',`<p><strong>New York State 2022 NAIP · RGB · native 0.6 m/pixel.</strong><br>Courtesy USDA-FSA; distributed by NOAA Digital Coast.</p><p>Statewide and county views use averaged photographic overviews cached from your downloaded TIFFs. Zoom 14 and above reads and reprojects the needed native windows. Sources stay in their original NAD83 UTM grids. Display downsampling changes what you see, not your search chips.</p><p><button id="prepare-overview" ${['starting','preparing'].includes(state.status?.overview?.state||'')?'disabled':''}>Prepare photographic overview</button></p><p>Cache: ${((state.status?.cache.bytes||0)/1e9).toFixed(2)} / ${((state.status?.cache.limit_bytes||5e9)/1e9).toFixed(2)} GB. The hard limit covers temporary cache writes. Set the limit in frontend/settings.json and relaunch.</p><p><button id="clear-cache">Clear frontend imagery cache…</button></p><p><label><input id="allow-partial" type="checkbox" ${state.allowPartial?'checked':''}> Allow published bundles from an idle interrupted collection</label></p><p class="muted">This does not permit queries while the worker or GPU is busy. Paused partial collections are already searchable. The frontend never sends pause signals.</p>`);$('#prepare-overview').addEventListener('click',()=>run(async()=>{await api('/api/overview/prepare',{confirm:'prepare-local-overview'});await refreshStatus();$<HTMLDialogElement>('#dialog').close();toast('Photographic overview preparation started. Native imagery browsing remains available.');}));$('#allow-partial').addEventListener('change',event=>{state.allowPartial=(event.target as HTMLInputElement).checked;});$('#clear-cache').addEventListener('click',()=>{openDialog('Clear frontend cache?', '<p>This deletes only generated frontend/cache/ tiles and previews. Original imagery, embeddings, bookmarks, and saved searches stay at their existing paths.</p><p><button id="confirm-cache" class="primary">Clear frontend cache</button></p>');$('#confirm-cache').addEventListener('click',()=>run(async()=>{await api('/api/cache/clear',{confirm:'frontend-cache-only'});$<HTMLDialogElement>('#dialog').close();await refreshStatus();toast('Frontend imagery cache cleared. Visible tiles will be regenerated when needed.');}));});});
document.addEventListener('keydown',event=>{if(event.ctrlKey&&event.key.toLowerCase()==='k'){event.preventDefault();$('#query').focus();}});
for(const side of ['left','right']){
  const handle=$(`.resize-handle.${side}`);let dragging=false;
  handle.addEventListener('pointerdown',event=>{dragging=true;handle.setPointerCapture(event.pointerId);});
  handle.addEventListener('pointermove',event=>{if(!dragging)return;const maximum=Math.min(390,innerWidth*.3);const width=Math.max(215,Math.min(maximum,side==='left'?event.clientX:innerWidth-event.clientX));document.documentElement.style.setProperty(side==='left'?'--left':'--right',`${width}px`);map.invalidateSize({pan:false});});
  handle.addEventListener('pointerup',()=>{dragging=false;map.invalidateSize();});handle.addEventListener('pointercancel',()=>{dragging=false;});
  handle.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;event.preventDefault();const variable=side==='left'?'--left':'--right';const current=parseInt(getComputedStyle(document.documentElement).getPropertyValue(variable));document.documentElement.style.setProperty(variable,`${Math.max(215,Math.min(390,current+(event.key==='ArrowRight'?15:-15)*(side==='left'?1:-1)))}px`);map.invalidateSize();});
}
new ResizeObserver(()=>map.invalidateSize({pan:false})).observe($('#map'));
async function start(){
  state.config=await api<Configuration>('/api/config');
  if(state.config.imagery){
    const extent=geographicBounds(state.config.imagery.bounds);
    (tiles.options as L.GridLayerOptions).bounds=extent.pad(.025);
    const view=new URLSearchParams(location.search).get('view')?.split(',').map(Number);
    if(view?.length===3&&view.every(Number.isFinite)&&extent.contains([view[1],view[0]])&&view[2]>=12&&view[2]<=20)map.setView([view[1],view[0]],view[2]);
    else fitState();
    tiles.addTo(map);detailLevel();
  }
  else notice(`Imagery unavailable: ${state.config.imagery_error}`);
  state.bookmarks=await api<Chip[]>('/api/bookmarks');await refreshStatus();
  window.setInterval(()=>run(refreshStatus),5000);
}
run(start);
