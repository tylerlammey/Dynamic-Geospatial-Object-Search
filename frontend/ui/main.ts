import L from 'leaflet';
import type {Chip, Configuration, DetectionBox, DetectionOptions, ResultSet, SearchProgress, Status} from './types';
import {api, authenticatedUrl as url, token} from './api';
import {createLocalTiles} from './local-tiles';
import './style.css';

const $=<T extends HTMLElement=HTMLElement>(selector:string)=>{const value=document.querySelector<T>(selector);if(!value)throw new Error(`Missing UI control: ${selector}`);return value;};
const escape=(value:unknown)=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const state:{results:Chip[];session:ResultSet|null;selected:Chip|null;page:number;coverage:boolean;status:Status|null;config:Configuration|null;bookmarks:Chip[];task:string|null;allowPartial:boolean}={results:[],session:null,selected:null,page:0,coverage:false,status:null,config:null,bookmarks:[],task:null,allowPartial:false};
const PAGE=20;
let baseOnly=false;
let showTileOutlines=false;
let showPartialBoxes=false;
const detection:DetectionOptions={enabled:true,box_threshold:0.30,max_candidates:256};
const scoped=(route:string)=>`${route}${route.includes('?')?'&':'?'}base_only=${baseOnly?1:0}`;
let modelAction=false;
let gapOverviewRequested=false;
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
const detectionPane=map.createPane('object-boxes');
detectionPane.style.zIndex='420';
detectionPane.style.pointerEvents='none';
const detectionRenderer=L.svg({pane:'object-boxes'});
const tileOutlineRenderer=L.svg();

const tiles = createLocalTiles({
  route: coords => {
    const revision = `${state.status?.overview?.epoch || 0}-${state.status?.overview?.ready ? 'ready' : 'preparing'}`;
    return url(scoped(`/api/tiles/${coords.z}/${coords.x}/${coords.y}.png?display=${revision}`));
  },
  report: message => { $('#tile-status').textContent = message; },
});

function geographicBounds(bounds:[number,number,number,number]):L.LatLngBounds{return L.latLngBounds([bounds[1],bounds[0]],[bounds[3],bounds[2]]);}
function fitState(){const bounds=state.config?.imagery?.bounds;if(bounds)map.fitBounds(geographicBounds(bounds),{padding:[25,40],maxZoom:8});}
function chipBounds(row:Chip){return geographicBounds([row.west,row.south,row.east,row.north]);}
function chipKey(row:Chip){return row.embedding_id===null?row.chip_id:`${row.dataset_id||'nys_naip_2022_rgb060'}:${row.shard_id}:${row.embedding_id}`;}
function strongestBox(row:Chip):DetectionBox|undefined{
  return row.detection?.predictions.reduce<DetectionBox|undefined>((best,box)=>
    !best||box.score>best.score?box:best,undefined);
}
function touchesTileEdge(row:Chip,prediction:DetectionBox){
  const [x1,y1,x2,y2]=prediction.box;
  // Older saved predictions do not include the detector's edge flag.
  return prediction.touches_tile_edge??(x1<=2||y1<=2||x2>=row.pixel_width-2||y2>=row.pixel_height-2);
}
function isLargePartialBox(row:Chip,prediction:DetectionBox){
  const [x1,y1,x2,y2]=prediction.box;
  return touchesTileEdge(row,prediction)&&(x2-x1)*(y2-y1)>=0.25*row.pixel_width*row.pixel_height;
}
function visibleBoxes(row:Chip){
  return (row.detection?.predictions||[]).filter(box=>showPartialBoxes||!isLargePartialBox(row,box));
}
function markerLocation(row:Chip){
  const ring=strongestBox(row)?.geometry?.coordinates[0];
  if(ring?.length){
    return L.latLngBounds(ring.map(([lon,lat])=>L.latLng(lat,lon))).getCenter();
  }
  return L.latLng(row.centroid_lat,row.centroid_lon);
}
function previewUrl(row:Chip,context=false){return url(`/api/preview?source=${encodeURIComponent(row.source_uri)}&x=${row.pixel_x}&y=${row.pixel_y}&context=${context?1:0}`);}
function detailLevel(){
  const zoom=map.getZoom();
  $('#detail-level').textContent=zoom<=10?'Statewide aerial overview':zoom<14?'Aerial overview':zoom<18?'Downloaded aerial imagery':'Native-detail aerial imagery';
  if(state.config?.synthetic)$('#detail-level').textContent='Generated synthetic imagery';
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
  const detectionResults=!!state.session?.detection?.enabled||state.results.some(row=>!!row.detection);
  const selectedHasBoxes=!!selected?.detection?.predictions.length;
  const outlineToggle=$<HTMLButtonElement>('#tile-outlines');
  outlineToggle.classList.toggle('hidden',!(detectionResults||selectedHasBoxes));
  outlineToggle.setAttribute('aria-pressed',String(showTileOutlines));
  if(selected&&(!selectedHasBoxes||showTileOutlines)){
    L.geoJSON(selected.geometry,{style:{color:'#ffc951',weight:3,fillColor:'#ffc951',fillOpacity:.08,
      ...(selectedHasBoxes?{renderer:tileOutlineRenderer,className:'detection-tile-outline'}:{})}}).addTo(selectedLayer);
  }
  if(selected&&map.getZoom()>=14){
    for(const prediction of visibleBoxes(selected)){
      L.geoJSON(prediction.geometry,{
        style:{color:'#33e6f4',weight:2,fill:false,interactive:false,renderer:detectionRenderer,
          ...(touchesTileEdge(selected,prediction)?{dashArray:'6 4'}:{})},
      }).addTo(selectedLayer);
    }
  }
  const paddedBounds=map.getBounds().pad(.15);
  const positions=new Map<Chip,L.LatLng>();
  const visible=state.results.filter(row=>{
    const position=markerLocation(row);
    if(!paddedBounds.contains(position))return false;
    positions.set(row,position);
    return true;
  });
  if(map.getZoom()>=14&&(!detectionResults||showTileOutlines)){
    for(const row of visible){
      if(detectionResults&&selected&&chipKey(row)===chipKey(selected))continue;
      L.geoJSON(row.geometry,{
        style:{color:chipKey(row)===(selected&&chipKey(selected))?'#ffc951':'#f6faf4',weight:1,fillOpacity:.02,
          ...(detectionResults?{renderer:tileOutlineRenderer,className:'detection-tile-outline'}:{})},
        onEachFeature:(_,layer)=>layer.on('click',event=>{L.DomEvent.stopPropagation(event);select(row);}),
      }).addTo(footprints);
    }
  }
  const grouped=new Map<string,Chip[]>();
  for(const row of visible){const pixel=map.latLngToContainerPoint(positions.get(row)!);const cell=map.getZoom()>=16?40:65;const key=`${Math.floor(pixel.x/cell)}:${Math.floor(pixel.y/cell)}`;const group=grouped.get(key)||[];group.push(row);grouped.set(key,group);}
  for(const rows of grouped.values()){
    const lon=rows.reduce((s,r)=>s+positions.get(r)!.lng,0)/rows.length;const lat=rows.reduce((s,r)=>s+positions.get(r)!.lat,0)/rows.length;
    const chosen=!!selected&&rows.some(r=>chipKey(r)===chipKey(selected));
    const offset=chosen&&selectedHasBoxes&&map.getZoom()>=14?24:0;
    const size=rows.length>1?36:29;
    const marker=L.marker([lat,lon],{icon:L.divIcon({className:`rank-marker${rows.length>1?' cluster':''}${chosen?' selected':''}`,html:String(rows.length>1?rows.length:rows[0].rank),iconSize:[size,size],iconAnchor:[size/2,size/2+offset]}),keyboard:true,title:rows.length>1?`${rows.length} nearby matches`:`Rank ${rows[0].rank}`}).addTo(overlays);
    marker.on('click',()=>{
      if(rows.length===1){select(rows[0]);return;}
      const popup=document.createElement('div');const title=document.createElement('strong');title.textContent=`${rows.length} nearby matches`;popup.append(title);
      const caption=document.createElement('p');caption.textContent='Every overlapping chip is retained.';popup.append(caption);
      const list=document.createElement('div');list.className='cluster-list';popup.append(list);
      for(const row of rows){const button=document.createElement('button');button.textContent=`#${row.rank}`;button.addEventListener('click',()=>{map.closePopup();select(row);});list.append(button);}
      const fit=document.createElement('button');fit.textContent='Zoom to these matches';fit.addEventListener('click',()=>{map.closePopup();map.fitBounds(L.latLngBounds(rows.map(row=>positions.get(row)!)),{padding:[50,70],maxZoom:17});});popup.append(fit);
      marker.bindPopup(popup).openPopup();
    });
  }
}
function renderResults(){
  $('#results-title').textContent=state.session?`“${state.session.query}”`:'Search matches';$('#result-count').textContent=String(state.results.length);
  $('#result-order').textContent=state.session?.detection?.enabled||state.results.some(row=>!!row.detection)
    ?'Sorted by highest box confidence. Similarity breaks ties.'
    :'Similarity ranks chips; it is not detection confidence.';
  if(!state.session){
    $('#result-list').innerHTML='<div class="empty-state"><strong>Your imagery is ready to explore.</strong><p>Run a text search or click downloaded imagery to inspect a native chip.</p></div>';
    $('.pagination').classList.add('hidden');$('#page-label').textContent='';return;
  }
  const start=state.page*PAGE;const page=state.results.slice(start,start+PAGE);
  $('#result-list').innerHTML=page.length?page.map(row=>{
    const best=strongestBox(row);
    const confidence=best?`<span class="score box-confidence">${best.score.toFixed(3)} box confidence</span>`:'';
    return `<button class="result-card${state.selected&&chipKey(row)===chipKey(state.selected)?' selected':''}" data-rank="${row.rank}" aria-pressed="${!!state.selected&&chipKey(row)===chipKey(state.selected)}"><img loading="lazy" src="${escape(previewUrl(row))}" alt="Native aerial chip, rank ${row.rank}"><span class="info"><span class="rank">${String(row.rank).padStart(2,'0')} · ${row.centroid_lat.toFixed(3)}° N</span>${confidence}<span class="score">${row.score!.toFixed(4)} similarity</span><span class="coords">${row.centroid_lon.toFixed(4)}, ${row.centroid_lat.toFixed(4)}</span></span></button>`;
  }).join(''):'<div class="empty-state"><strong>No matches.</strong><p>Try another phrase, and check the searched coverage.</p></div>';
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
  $('#selection .details div:first-child dd').textContent=state.config?.synthetic?'Synthetic generated RGB':row.release_year===2021?'Gap fill · 2021 release · RGB':'2022 NAIP · RGB';
  if(state.config?.synthetic)$('#selection .preview-label span:last-child').textContent='Generated imagery';
  if(row.detection?.predictions.length){
    const preview=$<HTMLImageElement>('#selection .selected-preview');
    const overlay=document.createElement('div');
    overlay.className='detection-preview';
    const title=state.config?.synthetic?'Mock detections':'Object detections';
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.classList.add('detection-boxes');
    svg.setAttribute('viewBox',`0 0 ${row.pixel_width} ${row.pixel_height}`);
    svg.setAttribute('role','img');
    svg.setAttribute('aria-label',`${title} on selected chip`);
    $('#selection .score-note').textContent='Tiles are ranked by highest box confidence. Similarity breaks ties.';
    const boxes=visibleBoxes(row);
    const hidden=row.detection.predictions.length-boxes.length;
    const partialCount=row.detection.predictions.filter(box=>isLargePartialBox(row,box)).length;
    for(const prediction of boxes){
      const [x1,y1,x2,y2]=prediction.box;
      const rectangle=document.createElementNS(svg.namespaceURI,'rect');
      for(const [key,value] of Object.entries({x:x1,y:y1,width:x2-x1,height:y2-y1})){
        rectangle.setAttribute(key,String(value));
      }
      if(touchesTileEdge(row,prediction))rectangle.classList.add('touches-tile-edge');
      svg.append(rectangle);
    }
    if(boxes.length){
      preview.before(overlay);
      overlay.append(preview,svg);
    }
    $('#selection .score-note').insertAdjacentHTML('afterend',
      `<div class="detection-detail"><strong>${title} (${row.detection.predictions.length})</strong>${boxes.length?`<ul>${boxes.map(box=>
        `<li>${escape(box.text_label||state.session?.query||'Detection')} <span>${box.score.toFixed(3)} confidence</span>${touchesTileEdge(row,box)?'<span class="edge-note">May extend beyond this tile</span>':''}</li>`).join('')}</ul>`:''}${partialCount?`<p class="muted">${hidden?`${hidden} large partial ${hidden===1?'box':'boxes'} hidden. `:''}Objects may continue beyond this tile.</p><button class="text-button" id="partial-box-toggle" aria-pressed="${showPartialBoxes}">Show large partial boxes</button>`:''}<p class="muted">Box confidence is a model score, not a calibrated probability.</p></div>`);
    if(partialCount){
      $('#partial-box-toggle').addEventListener('click',()=>{
        showPartialBoxes=!showPartialBoxes;
        renderSelection();renderOverlays();
        $('#partial-box-toggle').focus();
      });
    }
  }
  if(row.acquisition_date){const acquired=document.createElement('div');acquired.innerHTML=`<dt>Acquired</dt><dd>${escape(row.acquisition_date)}</dd>`;$('#selection .details').append(acquired);}
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
  if(results.detection?.enabled||results.results.some(row=>!!row.detection)){
    const confidence=new Map(results.results.map(row=>[row,strongestBox(row)?.score??-1]));
    state.results=[...results.results].sort((a,b)=>
      confidence.get(b)!-confidence.get(a)!||(b.score??-1)-(a.score??-1)
    ).map((row,index)=>({...row,rank:index+1}));
  }
  showTileOutlines=false;
  showPartialBoxes=false;
  $<HTMLInputElement>('#query').value=results.query;
  $('#search-scope').textContent=`Results searched ${results.searched_source_tiles.toLocaleString()} / ${results.total_source_tiles.toLocaleString()} source tiles${results.partial_collection?' · partial snapshot':''}`;
  if(results.search)$('#search-scope').textContent+=` · ${results.search.approximate?'Indexed':'Exact'} retrieval ${results.search.search_seconds.toFixed(2)}s`;
  if(results.detection?.enabled){
    const details=results.detection;
    $('#search-scope').textContent+=` · ${state.config?.synthetic?'Mock detections':'Tiles with detections'}: ${details.returned.toLocaleString()} · ${details.evaluated.toLocaleString()} tiles checked`;
    if(details.stop_reason==='candidate_limit')$('#search-scope').textContent+=` · tile limit reached (${details.max_candidates.toLocaleString()})`;
    else if(details.stop_reason==='low_detection_yield')$('#search-scope').textContent+=details.recent_detection_percent==null?' · few recent tiles matched':` · stopped at ${details.recent_detection_percent.toFixed(1)}% recent matches`;
    else if(details.stop_reason==='no_new_candidates')$('#search-scope').textContent+=' · no more candidates found';
  }
  state.selected=null;
  if(state.results.length)select(state.results[0],false);
  else{renderResults();renderSelection();renderOverlays();}
  showCoverage();
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
  if(state.config?.synthetic){
    $('#load-model').textContent=resident?'Mock model ready':'Enable mock model';
    $('.search-meta').textContent='Mock queries · no semantic relevance';
    $('#footer-worker').textContent='Synthetic development · no GPU model loaded';
  }
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
async function refreshStatus(){
  const before=state.status?.overview;state.status=await api<Status>(scoped('/api/status'));renderStatus();
  if(state.coverage&&!state.session?.searched_source_uris)showCoverage();const after=state.status.overview;
  if(before&&after&&(before.ready!==after.ready||before.epoch!==after.epoch)){tiles.redraw();if(after.ready)toast('Photographic overview is ready. Statewide and county views now use cached aerial imagery.');}
  if(!gapOverviewRequested&&!baseOnly&&(state.config?.imagery?.datasets?.length||0)>1&&!state.status.busy&&after?.state==='idle'&&!after.ready){
    gapOverviewRequested=true;await api('/api/overview/prepare',{confirm:'prepare-local-overview',base_only:false});
    toast('Preparing the gap-fill photographic overview. Existing state imagery remains visible.');
  }
}
async function pollModel(){
  await refreshStatus();
  if(['starting','loading'].includes(state.status?.model?.state||''))window.setTimeout(()=>run(pollModel),1000);
  else if(state.status?.model?.state==='ready')toast(state.config?.synthetic?'Mock model enabled. No GPU model was loaded.':'GeoRSCLIP is ready on the GPU. Searches will reuse this model.');
}
$('#load-model').addEventListener('click',()=>run(async()=>{
  if(modelAction)return;modelAction=true;renderStatus();
  try{await api('/api/model/load',{allow_partial:state.allowPartial});await pollModel();}
  finally{modelAction=false;renderStatus();}
}));
$('#unload-model').addEventListener('click',()=>run(async()=>{
  if(modelAction)return;modelAction=true;renderStatus();
  try{await api('/api/model/unload',{});await refreshStatus();toast(state.config?.synthetic?'Mock model disabled.':'Model unloaded; the GPU is available again.');}
  finally{modelAction=false;renderStatus();}
}));
async function pollSearch(identifier:string){
  const progress=await api<SearchProgress>(`/api/search-progress?id=${identifier}`);
  if(progress.state==='complete'){state.task=null;$('#progress').classList.add('hidden');$<HTMLButtonElement>('#search-button').textContent='Search imagery';await loadResults(progress.result_id!);await refreshStatus();toast(`Search finished: ${state.results.length} results from a fixed published snapshot.`);return;}
  if(progress.state==='error'){state.task=null;$('#progress').classList.add('hidden');$<HTMLButtonElement>('#search-button').textContent='Search imagery';renderStatus();throw new Error(progress.error);}
  const scanned=progress.scanned_sources||0;const total=progress.searched_source_tiles||state.status?.published_sources||1;
  if(progress.state==='loading_detector'){
    notice('Loading LLMDet on the GPU…');
    $('#progress i').style.width='2%';
  }else if(progress.state==='detecting'){
    const accepted=progress.accepted_candidates??0;
    const checked=progress.evaluated_candidates??0;
    const limit=progress.max_candidates??detection.max_candidates;
    const recent=progress.recent_detection_percent;
    notice(`Detecting objects: ${accepted.toLocaleString()} matching tiles · ${checked.toLocaleString()} checked (limit ${limit.toLocaleString()}) · round ${progress.retrieval_round??1}${recent==null?'':` · ${recent.toFixed(0)}% of recent tiles matched`}.`);
    $('#progress i').style.width=`${Math.max(2,Math.min(100,checked/limit*100))}%`;
  }else{
    notice(progress.state==='encoding'?'Encoding text with the pinned GeoRSCLIP model…':progress.state==='loading_index'?'Loading the local search index…':progress.state==='finishing'?'Finalizing results…':`Searching published coverage: ${scanned.toLocaleString()} of ${total.toLocaleString()} source tiles. You can keep exploring the map.`);
    $('#progress i').style.width=`${Math.max(2,scanned/total*100)}%`;
  }
  window.setTimeout(()=>run(()=>pollSearch(identifier)),750);
}
function updateSearchMode(){
  detection.enabled=$<HTMLInputElement>('#detect-objects').checked;
  $('#fixed-result-count').classList.toggle('hidden',detection.enabled);
  $<HTMLInputElement>('#top-k').disabled=detection.enabled;
}
$('#detect-objects').addEventListener('change',updateSearchMode);
updateSearchMode();
$('#search-form').addEventListener('submit',event=>{event.preventDefault();run(async()=>{
  if(state.task)return;
  const mode=$<HTMLSelectElement>('#search-mode').value;
  const result=await api<{task_id:string}>('/api/search',{query:$<HTMLInputElement>('#query').value,...(detection.enabled?{}:{top_k:Number($<HTMLInputElement>('#top-k').value)}),allow_partial:state.allowPartial,include_supplements:!baseOnly,search_backend:mode==='exact'?'exact':'auto',search_effort:mode==='thorough'?'thorough':'balanced',detection});
  state.task=result.task_id;renderStatus();$('#progress').classList.remove('hidden');$<HTMLButtonElement>('#search-button').textContent='Searching…';await pollSearch(result.task_id);
});});
$('#home').addEventListener('click',fitState);$('#zoom-selection').addEventListener('click',()=>state.selected?select(state.selected):toast('Click downloaded imagery to select a chip.'));
$('#tile-outlines').addEventListener('click',()=>{showTileOutlines=!showTileOutlines;renderOverlays();});
$('#fit-matches').addEventListener('click',()=>state.results.length?map.fitBounds(L.latLngBounds(state.results.map(markerLocation)),{padding:[40,75],maxZoom:17}):toast('Run a search or open saved results first.'));
$('#coverage-toggle').addEventListener('click',()=>{state.coverage=!state.coverage;$('#coverage-toggle').setAttribute('aria-pressed',String(state.coverage));if(!state.coverage)$('#coverage-toggle').textContent='Show search coverage';showCoverage();});
$('#page-prev').addEventListener('click',()=>{state.page=Math.max(0,state.page-1);renderResults();});$('#page-next').addEventListener('click',()=>{state.page++;renderResults();});
map.on('moveend zoomend',()=>{renderOverlays();detailLevel();});
map.on('mousemove',(event:L.LeafletMouseEvent)=>{$('#coordinates').textContent=`Lon ${event.latlng.lng.toFixed(5)} / Lat ${event.latlng.lat.toFixed(5)}`;});
let inspectController:AbortController|null=null;
map.on('click',(event:L.LeafletMouseEvent)=>{run(async()=>{
  inspectController?.abort();inspectController=new AbortController();
  const response=await fetch(scoped(`/api/inspect?lon=${event.latlng.lng}&lat=${event.latlng.lat}`),{headers:{Authorization:`Bearer ${token}`},signal:inspectController.signal});
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
$('#settings').addEventListener('click',()=>{
  const paragraph=document.createElement('p');
  paragraph.innerHTML=`<label><input id="base-only" type="checkbox" ${baseOnly?'checked':''} ${state.task?'disabled':''}> Use only the original 2022 collection for imagery and new searches</label><br><span class="muted">Gap-fill imagery comes from the 2021 release. Acquisition dates are shown in the chip inspector.</span>`;
  // Registered before the original Settings listener; append after it has
  // replaced the dialog contents and bound its original controls.
  queueMicrotask(()=>{
    $('#dialog-content').append(paragraph);
    const detectorSettings=document.createElement('div');
    detectorSettings.innerHTML=`<p><label>Detection box score threshold <input id="box-threshold" type="number" min="0" max="1" step="any" value="${detection.box_threshold}" required></label><br><span class="muted">Raise the threshold for stricter filtering. The score is not a calibrated probability.</span></p><p><label>Maximum tiles to check <input id="candidate-limit" type="number" min="1" max="5000" step="1" value="${detection.max_candidates}" required></label><br><span id="candidate-limit-note" class="muted"></span></p>`;
    $('#dialog-content').append(detectorSettings);
    const defaults=state.config?.detection_defaults;
    $('#candidate-limit-note').textContent=defaults
      ?`Object search shows each matching tile it finds. After checking ${defaults.min_candidates} tiles, it can stop if fewer than ${100*defaults.min_detection_rate}% of recent tiles match.`
      :'Object search shows each matching tile it finds. It can stop before this limit if few recent tiles match.';
    for(const [selector,key] of [['#box-threshold','box_threshold'],['#candidate-limit','max_candidates']] as const){
      const input=$<HTMLInputElement>(selector);
      input.addEventListener('change',()=>{
        if(input.reportValidity())detection[key]=input.valueAsNumber;
        else input.value=String(detection[key]);
      });
    }
    $('#base-only').addEventListener('change',event=>run(async()=>{
      baseOnly=(event.target as HTMLInputElement).checked;
      state.config=await api<Configuration>(scoped('/api/config'));
      state.results=[];state.session=null;state.selected=null;state.page=0;state.coverage=false;
      $('#search-scope').textContent='';$('#coverage-toggle').setAttribute('aria-pressed','false');
      $('#coverage-toggle').textContent='Show search coverage';map.closePopup();
      renderResults();renderSelection();renderOverlays();showCoverage();await refreshStatus();tiles.redraw();
      $<HTMLDialogElement>('#dialog').close();
      toast(baseOnly?'Using original 2022 imagery and embeddings only.':'Gap-fill imagery and available embeddings included.');
    }));
  });
});
$('#bookmarks').addEventListener('click',()=>run(async()=>{state.bookmarks=await api<Chip[]>('/api/bookmarks');openDialog('Bookmarked chips',state.bookmarks.length?`<div class="dialog-list">${state.bookmarks.map((r,i)=>`<button data-bookmark="${i}">${r.centroid_lon.toFixed(5)}, ${r.centroid_lat.toFixed(5)}<small>${escape(r.source_uri)}</small></button>`).join('')}</div>`:'<p>Select a native chip, then choose “Bookmark this chip.” Bookmarks are saved locally under frontend/state/.</p>');$('#dialog-content').querySelectorAll<HTMLButtonElement>('[data-bookmark]').forEach(button=>button.addEventListener('click',()=>{$<HTMLDialogElement>('#dialog').close();select(state.bookmarks[Number(button.dataset.bookmark)]);}));}));
$('#settings').addEventListener('click',()=>{const intro=state.config?.synthetic?`<p><strong>Synthetic development imagery · generated RGB patterns.</strong><br>${escape(state.config.attribution)}</p>`:'<p><strong>New York State 2022 NAIP · RGB · native 0.6 m/pixel.</strong><br>Courtesy USDA-FSA; distributed by NOAA Digital Coast.</p>';openDialog('Local imagery & cache',`${intro}<p>Statewide and county views use averaged photographic overviews cached from your downloaded TIFFs. Zoom 14 and above reads and reprojects the needed native windows. Sources stay in their original NAD83 UTM grids. Display downsampling changes what you see, not your search chips.</p><p><button id="prepare-overview" ${['starting','preparing'].includes(state.status?.overview?.state||'')?'disabled':''}>Prepare photographic overview</button></p><p>Cache: ${((state.status?.cache.bytes||0)/1e9).toFixed(2)} / ${((state.status?.cache.limit_bytes||5e9)/1e9).toFixed(2)} GB. The hard limit covers temporary cache writes. Set the limit in frontend/settings.json and relaunch.</p><p><button id="clear-cache">Clear frontend imagery cache…</button></p><p><label><input id="allow-partial" type="checkbox" ${state.allowPartial?'checked':''}> Allow published bundles from an idle interrupted collection</label></p><p class="muted">This does not permit queries while the worker or GPU is busy. Paused partial collections are already searchable. The frontend never sends pause signals.</p>`);$('#prepare-overview').addEventListener('click',()=>run(async()=>{await api('/api/overview/prepare',{confirm:'prepare-local-overview'});await refreshStatus();$<HTMLDialogElement>('#dialog').close();toast('Photographic overview preparation started. Native imagery browsing remains available.');}));$('#allow-partial').addEventListener('change',event=>{state.allowPartial=(event.target as HTMLInputElement).checked;});$('#clear-cache').addEventListener('click',()=>{openDialog('Clear frontend cache?', '<p>This deletes only generated frontend/cache/ tiles and previews. Original imagery, embeddings, bookmarks, and saved searches stay at their existing paths.</p><p><button id="confirm-cache" class="primary">Clear frontend cache</button></p>');$('#confirm-cache').addEventListener('click',()=>run(async()=>{await api('/api/cache/clear',{confirm:'frontend-cache-only'});$<HTMLDialogElement>('#dialog').close();await refreshStatus();toast('Frontend imagery cache cleared. Visible tiles will be regenerated when needed.');}));});});
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
  state.config=await api<Configuration>(scoped('/api/config'));
  if(state.config.detection_defaults){
    detection.box_threshold=state.config.detection_defaults.box_threshold;
    detection.max_candidates=state.config.detection_defaults.max_candidates;
  }
  if(state.config.synthetic){
    $('#synthetic-banner').classList.remove('hidden');
    $('#synthetic-banner').textContent='SYNTHETIC DEVELOPMENT · Generated imagery and mocked model/query results. Scores do not measure semantic relevance.';
    $('.brand small').textContent='Synthetic development workspace';
    $('.local-pill').textContent='SYNTHETIC IMAGERY';
    $('#load-model').title='Enable the development mock; no model or GPU inference';
    $('#unload-model').title='Disable the development mock';
    $('#unload-model').textContent='Disable mock model';
    $('.status-footer span:last-child').textContent='Generated pixels · synthetic 0.6 m/px grid';
    map.attributionControl.removeAttribution('Courtesy USDA-FSA; distributed by NOAA Digital Coast');
    map.attributionControl.addAttribution(escape(state.config.attribution));
  }
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
