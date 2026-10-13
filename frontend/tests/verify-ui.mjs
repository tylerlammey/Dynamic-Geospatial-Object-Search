// Live imagery UI verification, with intercepted example search results only.
// No GPU query is attempted while the embedding/GPU locks are held.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdir,readFile,writeFile,readdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';

const here=path.dirname(fileURLToPath(import.meta.url));const front=path.resolve(here,'..');const root=path.dirname(front);
const artifact=path.join(here,'artifacts');const profile=path.join(front,'.browser-profile',`ui-${Date.now()}`);const ready=path.join(front,'logs',`ui-test-${Date.now()}.json`);
for(const directory of [artifact,profile,path.join(front,'temp'),path.join(front,'logs')])await mkdir(directory,{recursive:true});
const env={...process.env,PYTHONDONTWRITEBYTECODE:'1',GDAL_PAM_ENABLED:'NO',TEMP:path.join(front,'temp'),TMP:path.join(front,'temp')};
const reuseService=process.argv.includes('--existing-service');
const service=reuseService?null:spawn(path.join(root,'.venv/Scripts/python.exe'),['-B',path.join(front,'service/server.py'),'--frontend',front,'--ready-file',ready],{windowsHide:true,env,stdio:['ignore','ignore','pipe']});
let serviceLog='';service?.stderr.on('data',chunk=>{serviceLog+=chunk.toString();});
let session;let edge;let pageSocket;let browserSocket;let origin;
let id=0;const pending=new Map();const runtimeErrors=[];const requestUrls=[];const tileRequests=[];const mockSearchRequests=[];const mockHistoryClearRequests=[];const mockModelRequests=[];let mockResults=null;let mockHistoryCleared=false;let mockStatus=null;let mockModelState='unloaded';let modelLoadingPolls=0;
function connection(url){return new Promise((resolve,reject)=>{const ws=new WebSocket(url);ws.onopen=()=>resolve(ws);ws.onerror=reject;});}
function request(ws,method,params={}){return new Promise((resolve,reject)=>{const requestId=++id;const timeout=setTimeout(()=>{pending.delete(requestId);reject(new Error(`Timeout ${method}`));},20000);pending.set(requestId,{resolve,reject,timeout});ws.send(JSON.stringify({id:requestId,method,params}));});}
function receive(event){const message=JSON.parse(event.data);if(message.id&&pending.has(message.id)){const p=pending.get(message.id);pending.delete(message.id);clearTimeout(p.timeout);message.error?p.reject(new Error(message.error.message)):p.resolve(message.result);}
  if(message.method==='Runtime.exceptionThrown')runtimeErrors.push(message.params.exceptionDetails);
  if(message.method==='Network.requestWillBeSent'){requestUrls.push(message.params.request.url);if(message.params.request.url.includes('/api/tiles/'))tileRequests.push(message.params.request.url);}
  if(message.method==='Fetch.requestPaused'){
    const data=message.params;
    if(mockResults&&data.request.url.endsWith('/api/model/load')){mockModelRequests.push(JSON.parse(data.request.postData));mockModelState='loading';modelLoadingPolls=0;void request(pageSocket,'Fetch.fulfillRequest',{requestId:data.requestId,responseCode:202,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify({state:'starting'})).toString('base64')});return;}
    if(mockResults&&data.request.url.endsWith('/api/model/unload')){mockModelState='unloaded';void request(pageSocket,'Fetch.fulfillRequest',{requestId:data.requestId,responseCode:200,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify({state:'unloaded'})).toString('base64')});return;}
    if(mockResults&&new URL(data.request.url).pathname==='/api/status'){if(mockModelState==='loading'&&++modelLoadingPolls>=3)mockModelState='ready';void request(pageSocket,'Fetch.fulfillRequest',{requestId:data.requestId,responseCode:200,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify({...mockStatus,busy:false,model:{state:mockModelState}})).toString('base64')});return;}
    if(data.request.url.includes('/api/search-progress?')&&mockResults){void request(pageSocket,'Fetch.fulfillRequest',{requestId:data.requestId,responseCode:200,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify({state:'complete',result_id:'0123456789abcdef0123456789abcdef'})).toString('base64')});}
    else if(data.request.url.endsWith('/api/search')&&data.request.method==='POST'&&mockResults){mockSearchRequests.push(JSON.parse(data.request.postData));void request(pageSocket,'Fetch.fulfillRequest',{requestId:data.requestId,responseCode:202,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify({task_id:'0123456789abcdef0123456789abcdef'})).toString('base64')});}
    else if(data.request.url.includes('/api/results?')&&mockResults){void request(pageSocket,'Fetch.fulfillRequest',{requestId:data.requestId,responseCode:200,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify(mockResults)).toString('base64')});}
    else if(data.request.url.endsWith('/api/history/clear')&&data.request.method==='POST'&&mockResults){mockHistoryClearRequests.push(JSON.parse(data.request.postData));mockHistoryCleared=true;void request(pageSocket,'Fetch.fulfillRequest',{requestId:data.requestId,responseCode:200,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify({removed_files:6,freed_bytes:2000000})).toString('base64')});}
    else if(data.request.url.includes('/api/history')&&mockResults){void request(pageSocket,'Fetch.fulfillRequest',{requestId:data.requestId,responseCode:200,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify(mockHistoryCleared?[]:[{id:'0123456789abcdef0123456789abcdef',query:'EXAMPLE TEST · baseball field',count:3,searched_source_tiles:250,total_source_tiles:3805,created_utc:Date.now()/1000}])).toString('base64')});}
    else void request(pageSocket,'Fetch.continueRequest',{requestId:data.requestId});
  }
}
async function evaluate(expression){const result=await request(pageSocket,'Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
async function click(selector){await evaluate(`document.querySelector(${JSON.stringify(selector)}).dispatchEvent(new MouseEvent('click',{bubbles:true}))`);}
async function waitFor(expression,timeout=15000){for(let elapsed=0;elapsed<timeout;elapsed+=100){if(await evaluate(expression))return;await delay(100);}throw new Error(`UI condition timed out: ${expression}`);}
async function screenshot(name){const image=await request(pageSocket,'Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(path.join(artifact,`${name}.png`),Buffer.from(image.data,'base64'));}
async function api(route,body){return fetch(origin+route,{method:body===undefined?'GET':'POST',headers:{Authorization:`Bearer ${session.token}`,...(body===undefined?{}:{'Content-Type':'application/json'})},body:body===undefined?undefined:JSON.stringify(body)});}

try{
  if(reuseService){
    for(const name of await readdir(path.join(front,'logs'))){
      if(!/^desktop-[a-f0-9-]+\.json$/.test(name))continue;
      try{const candidate=JSON.parse(await readFile(path.join(front,'logs',name),'utf8'));const response=await fetch(`http://127.0.0.1:${candidate.port}/api/status`,{headers:{Authorization:`Bearer ${candidate.token}`}});if(response.ok){session=candidate;break;}}catch{}
    }
  }else for(let attempt=0;attempt<600;attempt++){if(service.exitCode!==null)throw new Error(serviceLog);try{session=JSON.parse(await readFile(ready,'utf8'));break;}catch{}await delay(100);}
  assert.ok(session,'Service failed to start');origin=`http://127.0.0.1:${session.port}`;
  const status=await (await api('/api/status')).json();const config=await (await api('/api/config')).json();
  const history=await (await api('/api/history')).json();
  const cached=history.find(r=>r.id.startsWith('backend-')&&r.query==='baseball field')||history.find(r=>r.id.startsWith('backend-'));
  if(cached){const savedResponse=await api(`/api/results?id=${cached.id}`);assert.equal(savedResponse.status,200);const saved=await savedResponse.json();assert.ok(saved.results.length>0);assert.equal(saved.results[0].geometry.type,'Polygon');assert.equal(saved.run_fingerprint,status.run_fingerprint);}
  assert.equal(config.imagery.datasets?.find(dataset=>dataset.release_year===2022)?.sources||config.imagery.count,3805);
  assert.equal(config.imagery.count,config.sources.length);assert.equal((await fetch(origin+'/api/status')).status,403);
  assert.equal((await fetch(origin+'/api/status',{headers:{Authorization:`Bearer ${session.token}`,Origin:'https://example.com'}})).status,403);
  assert.equal((await api('/api/tiles/21/0/0.png')).status,400);
  const overviewChecks=[];
  const source=config.sources.find(s=>s.bounds[0]<-73.7562&&s.bounds[2]>-73.7562&&s.bounds[1]<42.6526&&s.bounds[3]>42.6526)||config.sources[0];
  const lon=(source.bounds[0]+source.bounds[2])/2;const lat=(source.bounds[1]+source.bounds[3])/2;
  const inspect=await api(`/api/inspect?lon=${lon}&lat=${lat}`);assert.equal(inspect.status,200);const chip=await inspect.json();
  assert.equal(chip.pixel_width,256);
  const preview=await api(`/api/preview?source=${encodeURIComponent(chip.source_uri)}&x=${chip.pixel_x}&y=${chip.pixel_y}`);assert.equal(preview.status,200);assert.equal(preview.headers.get('content-type'),'image/png');
  await writeFile(path.join(artifact,'actual-native-chip.png'),Buffer.from(await preview.arrayBuffer()));
  if(status.busy){const blocked=await api('/api/search',{query:'baseball field',top_k:3});assert.equal(blocked.status,409);assert.match((await blocked.json()).error,/Embedding is running|A frontend search is already running/);}
  edge=spawn(process.env.FRONTEND_EDGE_PATH||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',['--headless=new','--disable-gpu','--remote-debugging-port=0',`--user-data-dir=${profile}`,`--disk-cache-dir=${path.join(profile,'cache')}`,'--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-component-update','--disable-sync','--disable-extensions','about:blank'],{windowsHide:true,env,stdio:'ignore'});
  let port;
  for(let attempt=0;attempt<200;attempt++){try{port=Number((await readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0]);if(port)break;}catch{}await delay(100);}
  assert.ok(port,'Edge debugging endpoint unavailable');
  const targets=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();const version=await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  pageSocket=await connection(targets.find(t=>t.type==='page').webSocketDebuggerUrl);pageSocket.onmessage=receive;
  browserSocket=await connection(version.webSocketDebuggerUrl);browserSocket.onmessage=receive;
  await request(pageSocket,'Runtime.enable');await request(pageSocket,'Page.enable');await request(pageSocket,'Network.enable');
  await request(pageSocket,'Emulation.setDeviceMetricsOverride',{width:1440,height:960,deviceScaleFactor:1,mobile:false});
  if(process.argv.includes('--overview')){
    assert.equal(status.overview.ready,true,'Prepare photographic imagery before checking the overview');
    for(const [name,view] of [['statewide',null],['region','-75.8,43.2,10'],['county','-78.7203,42.9644,12'],['neighborhood','-78.7203,42.9644,14']]){
      const started=Date.now();
      await request(pageSocket,'Page.navigate',{url:`${origin}/${view?'?view='+view:''}#${session.token}`});
      await waitFor("document.getElementById('coverage-text')?.textContent.includes('source tiles searchable')");
      await waitFor("document.querySelectorAll('.leaflet-tile').length>=4 && [...document.querySelectorAll('.leaflet-tile')].every(i=>i.complete&&i.naturalWidth===256)",45000);
      await delay(200);await screenshot(`photographic-${name}`);
      overviewChecks.push({view:name,seconds:(Date.now()-started)/1000,loaded:await evaluate("document.querySelectorAll('.leaflet-tile-loaded').length")});
    }
    await writeFile(path.join(artifact,'overview-report.json'),JSON.stringify({passed:true,views:overviewChecks},null,2));
    console.log('Photographic overview views verified: '+JSON.stringify(overviewChecks));
  }
  await request(pageSocket,'Page.navigate',{url:`${origin}/?view=${lon},${lat},18#${session.token}`});
  await waitFor("document.querySelector('.leaflet-tile-loaded') && document.getElementById('coverage-text').textContent.includes('source tiles searchable')");
  await waitFor("[...document.querySelectorAll('.leaflet-tile')].filter(i=>i.complete&&i.naturalWidth===256).length>=4");
  await request(pageSocket,'Input.dispatchMouseEvent',{type:'mousePressed',x:Math.round(await evaluate("document.getElementById('map').getBoundingClientRect().left+document.getElementById('map').clientWidth/2")),y:Math.round(await evaluate("document.getElementById('map').getBoundingClientRect().top+document.getElementById('map').clientHeight/2")),button:'left',clickCount:1});
  await request(pageSocket,'Input.dispatchMouseEvent',{type:'mouseReleased',x:Math.round(await evaluate("document.getElementById('map').getBoundingClientRect().left+document.getElementById('map').clientWidth/2")),y:Math.round(await evaluate("document.getElementById('map').getBoundingClientRect().top+document.getElementById('map').clientHeight/2")),button:'left',clickCount:1});
  await waitFor("document.querySelector('.selected-preview')?.complete && document.querySelector('.selected-preview')?.naturalWidth===256");
  await screenshot('actual-imagery-desktop');
  assert.deepEqual(await evaluate("[...document.querySelector('#search-mode').options].map(o=>o.value)"),['balanced','thorough','exact']);
  assert.equal(await evaluate("document.getElementById('search-mode').value"),'balanced');
  await click('#settings');
  assert.equal(await evaluate("document.getElementById('dialog').open"),true);
  assert.ok(await evaluate("!!document.getElementById('prepare-overview') && !!document.getElementById('clear-cache') && !!document.getElementById('allow-partial')"));
  await click('#close-dialog');
  if(cached){
    await click('#history');await waitFor(`!!document.querySelector('[data-history="${cached.id}"]')`);await click(`[data-history="${cached.id}"]`);
    await waitFor("document.querySelectorAll('.result-card').length>0");await click('.result-card');
    await waitFor("document.querySelector('.selected-preview')?.complete && document.querySelector('.selected-preview')?.naturalWidth===256");
    await delay(700);await screenshot('actual-cached-search');
  }
  const countBefore=tileRequests.length;await click('.leaflet-control-zoom-out');await delay(1000);await click('.leaflet-control-zoom-in');await delay(1000);assert.ok(tileRequests.length>countBefore,'Zoom must request another imagery level');
  await click('#coverage-toggle');assert.equal(await evaluate("document.getElementById('coverage-toggle').getAttribute('aria-pressed')"),'true');await click('#coverage-toggle');
  mockResults={query:'EXAMPLE TEST · baseball field',run_fingerprint:status.run_fingerprint,results:[{...chip,rank:1,score:.3184,embedding_id:1},{...chip,chip_id:chip.chip_id+'-2',rank:2,score:.3147,embedding_id:2},{...chip,chip_id:chip.chip_id+'-3',rank:3,score:.311,embedding_id:3}],searched_source_tiles:250,total_source_tiles:3805,searched_chip_count:1000,searched_source_uris:[chip.source_uri],partial_collection:true,created_utc:Date.now()/1000};
  mockStatus={...status,overview:{...status.overview,ready:true}};
  await request(pageSocket,'Fetch.enable',{patterns:[{urlPattern:'*/api/history*'},{urlPattern:'*/api/results*'},{urlPattern:'*/api/search*'},{urlPattern:'*/api/model/*'},{urlPattern:'*/api/status*'}]});
  await click('#history');await waitFor("!!document.querySelector('[data-history]')");await click('[data-history]');
  await waitFor("document.querySelectorAll('.result-card').length===3");
  await click('[data-rank="2"]');await waitFor("document.querySelector('.selection-nav').textContent.includes('Rank 2')");
  assert.ok(await evaluate("document.querySelector('[data-rank=\"2\"]').getAttribute('aria-pressed')==='true'"));
  await waitFor("!!document.querySelector('.rank-marker.cluster')");
  await click('.rank-marker.cluster');assert.equal(await evaluate("document.querySelectorAll('.cluster-list button').length"),3);await click('.cluster-list button');
  await waitFor("document.querySelector('.selection-nav').textContent.includes('Rank 1')");await click('[data-rank="2"]');
  await click('#next');await waitFor("document.querySelector('.selection-nav').textContent.includes('Rank 3')");
  await click('#previous');await click('#context-preview');assert.ok(await evaluate("document.getElementById('dialog').open"));await click('#close-dialog');
  await click('#fit-matches');await delay(500);
  for(const mode of ['balanced','thorough','exact']){
    await evaluate(`document.getElementById('query').value='EXAMPLE TEST'; document.getElementById('search-mode').value=${JSON.stringify(mode)}; document.getElementById('search-form').requestSubmit()`);
    for(let attempts=0;mockSearchRequests.length<['balanced','thorough','exact'].indexOf(mode)+1&&attempts<100;attempts++)await delay(100);
    assert.equal(mockSearchRequests.at(-1)?.search_backend,mode==='exact'?'exact':'auto');
    assert.equal(mockSearchRequests.at(-1)?.search_effort,mode==='thorough'?'thorough':'balanced');
    assert.equal(mockSearchRequests.at(-1)?.include_supplements,true);
    await waitFor("!document.getElementById('search-button').disabled && document.querySelectorAll('.result-card').length===3");
  }
  await click('#settings');await waitFor("!!document.querySelector('#base-only')");
  await evaluate("document.getElementById('base-only').checked=true; document.getElementById('base-only').dispatchEvent(new Event('change',{bubbles:true}))");
  await waitFor("!document.getElementById('dialog').open");
  await evaluate("document.getElementById('query').value='EXAMPLE TEST'; document.getElementById('search-form').requestSubmit()");
  for(let attempts=0;mockSearchRequests.length<4&&attempts<100;attempts++)await delay(100);
  assert.equal(mockSearchRequests.at(-1)?.include_supplements,false);
  await waitFor("!document.getElementById('search-button').disabled && document.querySelectorAll('.result-card').length===3");
  await click('#settings');await waitFor("!!document.querySelector('#base-only')");
  await evaluate("document.getElementById('base-only').checked=false; document.getElementById('base-only').dispatchEvent(new Event('change',{bubbles:true}))");
  await waitFor("!document.getElementById('dialog').open");
  // Intercept the destructive request: real user history must never be cleared.
  await click('#load-model');await waitFor("document.getElementById('load-model').textContent==='Loading model…'");
  assert.ok(await evaluate("document.getElementById('search-button').disabled"),'Search must wait for preload');
  await waitFor("document.getElementById('load-model').textContent==='Model on GPU' && !document.getElementById('search-button').disabled");
  assert.deepEqual(mockModelRequests,[{allow_partial:false}]);
  assert.ok(await evaluate("!document.getElementById('unload-model').classList.contains('hidden')"));
  await click('#history');await waitFor("!!document.querySelector('#clear-history')");
  await click('#clear-history');await waitFor("!!document.querySelector('#confirm-history')");
  assert.equal(mockHistoryClearRequests.length,0,'Cleanup requires confirmation');
  await click('#confirm-history');await waitFor("document.getElementById('dialog-title').textContent==='Saved searches' && !document.querySelector('[data-history]')");
  assert.deepEqual(mockHistoryClearRequests,[{confirm:'all-search-history'}]);
  assert.equal(await evaluate("document.querySelectorAll('.result-card, .rank-marker, .selected-preview, .selection-nav').length"),0,'Current results, markers, and inspector must clear');
  assert.equal(await evaluate("document.getElementById('result-count').textContent"),'0');
  assert.equal(await evaluate("document.getElementById('query').value"),'');
  assert.equal(await evaluate("document.getElementById('search-scope').textContent"),'');
  assert.equal(await evaluate("document.getElementById('load-model').textContent"),'Model on GPU','Clearing history must leave the resident model loaded');
  await click('#close-dialog');
  await click('#unload-model');await waitFor("document.getElementById('load-model').textContent==='Load model to GPU' && !document.getElementById('load-model').disabled");
  assert.ok(await evaluate("document.getElementById('unload-model').classList.contains('hidden')"));
  // Wide and scaled viewport checks on actual imagery; intercepted ranks are
  // test fixtures only and do not persist or mutate any embedding bundle.
  for(const [width,height,label] of [[1093,720,'scaled'],[1920,1080,'large'],[900,700,'compact']]){
    await request(pageSocket,'Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});await delay(600);
    assert.ok(await evaluate("document.documentElement.scrollWidth<=innerWidth"),`Horizontal overflow at ${width}`);
    await screenshot(`actual-imagery-${label}`);
  }
  assert.equal(runtimeErrors.length,0,JSON.stringify(runtimeErrors));
  assert.ok(requestUrls.every(u=>u.startsWith(origin)||u.startsWith('data:')||u.startsWith('blob:')),'UI must not request online imagery or CDNs');
  await writeFile(path.join(artifact,'ui-report.json'),JSON.stringify({passed:true,browser:version.Browser,liveImagery:true,workerBusy:status.busy,coverage:status.published_sources,tileRequests:tileRequests.length,checks:['actual native chip','visible XYZ imagery','zoom level requests','read-only busy rejection','authentication and origin checks','selection and footprint','overlap retention','previous/next','context','snapshot coverage','4 viewports','no external resource requests'],runtimeErrors},null,2));
  console.log('Live imagery UI verified, including busy-worker rejection and intercepted result selection. No GPU model loaded.');
}catch(error){
  if(pageSocket?.readyState===WebSocket.OPEN){
    try{await screenshot('failure');await writeFile(path.join(artifact,'failure.json'),JSON.stringify({error:String(error),runtimeErrors,ui:await evaluate("({notice:document.getElementById('notice')?.textContent,status:document.getElementById('coverage-text')?.textContent,tiles:document.querySelectorAll('.leaflet-tile').length,loaded:document.querySelectorAll('.leaflet-tile-loaded').length})")},null,2));}catch{}
  }
  throw error;
}finally{
  if(browserSocket?.readyState===WebSocket.OPEN){try{await request(browserSocket,'Browser.close');}catch{}browserSocket.close();}pageSocket?.close();
  if(edge&&edge.exitCode===null)edge.kill();
  if(session&&!reuseService){try{await api('/api/shutdown',{});}catch{}}
  if(service&&service.exitCode===null){await Promise.race([new Promise(resolve=>service.once('exit',resolve)),delay(7000)]);if(service.exitCode===null)service.kill();}
  await writeFile(path.join(artifact,'ui-service.log'),serviceLog);
}
