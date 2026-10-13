// Exercise the built UI with real local imagery and deliberately stalled tiles.
// A read-only proxy lets this run alongside the user's open desktop app.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {mkdir,readFile,readdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';

const front=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const root=path.dirname(front),artifact=path.join(front,'tests/artifacts');
const stamp=Date.now(),ready=path.join(front,'logs',`zoom-test-${stamp}.json`);
const profile=path.join(front,'.browser-profile',`zoom-${stamp}`);
for(const folder of [artifact,profile,path.join(front,'logs'),path.join(front,'temp')])await mkdir(folder,{recursive:true});
const env={...process.env,PYTHONDONTWRITEBYTECODE:'1',GDAL_PAM_ENABLED:'NO',TEMP:path.join(front,'temp'),TMP:path.join(front,'temp')};
let session,service,serviceLog='',edge,proxy,pageSocket,browserSocket,origin;
let hold=false,requestId=0;const gates=new Set(),pending=new Map(),runtimeErrors=[],timings=[],checks=[];
async function findSession(){
  for(const name of await readdir(path.join(front,'logs'))){
    if(!/^desktop-[a-f0-9-]+\.json$/.test(name))continue;
    try{
      const candidate=JSON.parse(await readFile(path.join(front,'logs',name),'utf8'));
      const response=await fetch(`http://127.0.0.1:${candidate.port}/api/status`,{headers:{Authorization:`Bearer ${candidate.token}`},signal:AbortSignal.timeout(2500)});
      if(response.ok)return candidate;
    }catch{}
  }
  service=spawn(path.join(root,'.venv/Scripts/python.exe'),['-B',path.join(front,'service/server.py'),'--frontend',front,'--ready-file',ready],{windowsHide:true,env,stdio:['ignore','ignore','pipe']});
  service.stderr.on('data',chunk=>{serviceLog+=chunk;});
  for(let attempt=0;attempt<900;attempt++){
    if(service.exitCode!==null)throw new Error(serviceLog);
    try{return JSON.parse(await readFile(ready,'utf8'));}catch{}
    await delay(100);
  }
  throw new Error('Local imagery service did not start');
}
const connect=url=>new Promise((resolve,reject)=>{const ws=new WebSocket(url);ws.onopen=()=>resolve(ws);ws.onerror=reject;});
function command(ws,method,params={}){
  return new Promise((resolve,reject)=>{
    const id=++requestId,timer=setTimeout(()=>{pending.delete(id);reject(new Error(`Timeout: ${method}`));},20000);
    pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));
  });
}
function receive(event){
  const message=JSON.parse(event.data),callback=pending.get(message.id);
  if(callback){clearTimeout(callback.timer);pending.delete(message.id);message.error?callback.reject(new Error(message.error.message)):callback.resolve(message.result);}
  if(message.method==='Runtime.exceptionThrown')runtimeErrors.push(message.params.exceptionDetails);
}
async function evaluate(expression){
  const value=await command(pageSocket,'Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(value.exceptionDetails)throw new Error(JSON.stringify(value.exceptionDetails));return value.result.value;
}
async function waitFor(expression,timeout=45000){
  const deadline=Date.now()+timeout;
  while(Date.now()<deadline){if(await evaluate(expression))return;await delay(100);}
  throw new Error(`Condition timed out: ${expression}`);
}
async function screenshot(name){const value=await command(pageSocket,'Page.captureScreenshot',{format:'png'});await writeFile(path.join(artifact,`${name}.png`),Buffer.from(value.data,'base64'));}
function release(){hold=false;for(const resolve of gates)resolve();gates.clear();}
async function stable(){
  await waitFor("document.querySelectorAll('.leaflet-tile-loaded').length>=4 && document.getElementById('tile-status').textContent===''");
  await delay(350);
}
async function transition(name,selector,steps=1){
  await stable();
  await evaluate("window.zoomPrevious=[...document.querySelectorAll('.leaflet-tile-loaded')]; window.zoomPreviousWidth=zoomPrevious[0].getBoundingClientRect().width");
  // Hold every replacement, including obsolete requests during rapid zooms.
  hold=true;
  const started=Date.now();
  for(let step=0;step<steps;step++){
    await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);await delay(350);
  }
  await waitFor('document.querySelectorAll(\'.leaflet-tile:not(.leaflet-tile-loaded)\').length>0',5000);
  // Longer than the fade/zoom timers: replacements are still held back.
  await delay(650);
  const before=await evaluate(`(()=>{
    const map=document.getElementById('map').getBoundingClientRect(),x=map.x+map.width/2,y=map.y+map.height/2;
    const retained=zoomPrevious.filter(image=>image.isConnected&&image.complete&&image.naturalWidth===256&&Number(getComputedStyle(image).opacity)>0);
    const covers=image=>{const r=image.getBoundingClientRect();return r.left<=x&&r.right>=x&&r.top<=y&&r.bottom>=y;};
    return {retained:retained.length,centerCovered:retained.some(covers),width:retained[0]?.getBoundingClientRect().width,previousWidth:zoomPreviousWidth,totalTiles:document.querySelectorAll('.leaflet-tile').length};
  })()`);
  await screenshot(`zoom-${name}-pending`);
  assert.ok(before.retained>0,`${name}: loaded imagery was removed before replacement tiles arrived`);
  assert.ok(before.centerCovered,`${name}: existing imagery must still cover the map center`);
  assert.ok(selector.endsWith('zoom-in')?before.width>before.previousWidth:before.width<before.previousWidth,`${name}: existing imagery must scale with the map`);
  release();await stable();
  const after=await evaluate("({old:zoomPrevious.filter(image=>image.isConnected).length,tiles:document.querySelectorAll('.leaflet-tile').length,loaded:document.querySelectorAll('.leaflet-tile-loaded').length})");
  assert.equal(after.old,0,`${name}: old imagery must be pruned after replacements fade in`);
  assert.ok(after.tiles<80,`${name}: retained imagery must remain bounded`);
  checks.push({name,steps,pending:before,settled:after,seconds:(Date.now()-started)/1000});
}

try{
  session=await findSession();
  proxy=createServer(async(request,response)=>{
    try{
      const route=new URL(request.url,'http://127.0.0.1');
      if(route.pathname.startsWith('/api/')){
        if(request.method!=='GET'){response.writeHead(405);response.end();return;}
        const supplied=request.headers.authorization?.replace(/^Bearer /,'')||route.searchParams.get('token');
        if(supplied!==session.token||request.headers.origin&&request.headers.origin!==origin){response.writeHead(403);response.end();return;}
        const tile=route.pathname.match(/^\/api\/tiles\/(\d+)\//);
        if(tile&&hold){
          await new Promise(resolve=>{gates.add(resolve);response.once('close',()=>{gates.delete(resolve);resolve();});});
          if(response.destroyed)return;
        }
        const started=Date.now(),controller=new AbortController();response.once('close',()=>controller.abort());
        const result=await fetch(`http://127.0.0.1:${session.port}${request.url}`,{headers:{Authorization:`Bearer ${session.token}`},signal:controller.signal});
        const blob=Buffer.from(await result.arrayBuffer());
        if(tile)timings.push({zoom:Number(tile[1]),milliseconds:Date.now()-started,status:result.status});
        response.writeHead(result.status,{'Content-Type':result.headers.get('content-type')||'application/json','Cache-Control':'no-store'});response.end(blob);return;
      }
      const relative=route.pathname==='/'?'index.html':decodeURIComponent(route.pathname.slice(1));
      const build=path.join(front,'build'),file=path.resolve(build,relative);
      if(!file.startsWith(build+path.sep)){response.writeHead(403);response.end();return;}
      const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.svg':'image/svg+xml'};
      const blob=await readFile(file);response.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'});response.end(blob);
    }catch(error){if(!response.destroyed){response.writeHead(error.name==='AbortError'?499:500);response.end(String(error));}}
  });
  await new Promise(resolve=>proxy.listen(0,'127.0.0.1',resolve));origin=`http://127.0.0.1:${proxy.address().port}`;
  edge=spawn(process.env.FRONTEND_EDGE_PATH||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',['--headless=new','--disable-gpu','--remote-debugging-port=0',`--user-data-dir=${profile}`,'--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-extensions','about:blank'],{windowsHide:true,env,stdio:'ignore'});
  let port;for(let attempt=0;attempt<250;attempt++){try{port=Number((await readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0]);if(port)break;}catch{}await delay(100);}
  assert.ok(port,'Headless Edge did not start');
  const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json(),version=await(await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  pageSocket=await connect(targets.find(target=>target.type==='page').webSocketDebuggerUrl);pageSocket.onmessage=receive;
  browserSocket=await connect(version.webSocketDebuggerUrl);browserSocket.onmessage=receive;
  await command(pageSocket,'Runtime.enable');await command(pageSocket,'Page.enable');await command(pageSocket,'Network.enable');
  await command(pageSocket,'Network.setCacheDisabled',{cacheDisabled:true});
  await command(pageSocket,'Emulation.setDeviceMetricsOverride',{width:1440,height:960,deviceScaleFactor:1,mobile:false});
  await command(pageSocket,'Page.navigate',{url:`${origin}/?view=-73.7562,42.6526,13#${session.token}`});
  await waitFor("document.getElementById('coverage-text').textContent.includes('source tiles searchable')");
  await transition('in','.leaflet-control-zoom-in');
  await transition('out','.leaflet-control-zoom-out');
  await transition('rapid-in','.leaflet-control-zoom-in',2);
  await transition('rapid-out','.leaflet-control-zoom-out',2);
  assert.equal(runtimeErrors.length,0,JSON.stringify(runtimeErrors));
  const report={passed:true,checks,tileTimings:timings,borrowedService:!service,noGpuInference:true};
  await writeFile(path.join(artifact,'zoom-report.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({passed:true,checks,tileRequests:timings.length,borrowedService:!service},null,2));
}catch(error){
  await writeFile(path.join(artifact,'zoom-failure.json'),JSON.stringify({error:String(error),checks,runtimeErrors},null,2));throw error;
}finally{
  release();
  if(browserSocket?.readyState===WebSocket.OPEN){try{await command(browserSocket,'Browser.close');}catch{}browserSocket.close();}pageSocket?.close();
  if(edge&&edge.exitCode===null)edge.kill();
  if(proxy){proxy.closeAllConnections();await new Promise(resolve=>proxy.close(resolve));}
  if(service&&session){try{await fetch(`http://127.0.0.1:${session.port}/api/shutdown`,{method:'POST',headers:{Authorization:`Bearer ${session.token}`,'Content-Type':'application/json'},body:'{}'});}catch{}}
  if(service&&service.exitCode===null){await Promise.race([new Promise(resolve=>service.once('exit',resolve)),delay(7000)]);if(service.exitCode===null)service.kill();}
  await writeFile(path.join(artifact,'zoom-service.log'),serviceLog);
}
