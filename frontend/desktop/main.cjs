const {app,BrowserWindow,dialog}=require('electron');
const {spawn}=require('node:child_process');
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');

const frontend=path.resolve(process.env.NYS_FRONTEND_ROOT || (app.isPackaged?path.join(path.dirname(process.execPath),'../..'):path.join(__dirname,'..')));
const repository=path.dirname(frontend);
for(const directory of ['logs','temp','.browser-profile/desktop'])fs.mkdirSync(path.join(frontend,directory),{recursive:true});
app.setPath('userData',path.join(frontend,'.browser-profile/desktop'));
app.setPath('sessionData',path.join(frontend,'.browser-profile/desktop'));
app.setPath('logs',path.join(frontend,'logs'));
app.setPath('crashDumps',path.join(frontend,'logs/crashes'));
app.disableHardwareAcceleration();
app.commandLine.appendSwitch('disable-background-networking');
app.commandLine.appendSwitch('disable-component-update');
let service;
let endpoint;
let window;
let closing=false;
let log;
const ownedReady=path.join(frontend,'logs',`desktop-${crypto.randomUUID()}.json`);
function launchLog(message){fs.appendFileSync(path.join(frontend,'logs/desktop-launch.log'),`${new Date().toISOString()} ${message}\n`);}
function showWindow(){if(window&&!window.isDestroyed()){if(window.isMinimized())window.restore();window.show();window.focus();}}

async function stopService(){
  if(!service||service.exitCode!==null)return;
  try{if(endpoint)await fetch(`http://127.0.0.1:${endpoint.port}/api/shutdown`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${endpoint.token}`},body:'{}',signal:AbortSignal.timeout(3000)});}catch{}
  await new Promise(resolve=>{if(service.exitCode!==null)return resolve();const timer=setTimeout(()=>{service.kill();resolve();},8000);service.once('exit',()=>{clearTimeout(timer);resolve();});});
  log?.end();
  if(fs.existsSync(ownedReady))fs.unlinkSync(ownedReady);
}

async function start(){
  launchLog(`Starting ${app.isPackaged?'packaged':'checkout'} desktop`);
  const python=process.env.NYS_FRONTEND_PYTHON || path.join(repository,'.venv/Scripts/python.exe');
  const server=app.isPackaged?path.join(process.resourcesPath,'service/server.py'):path.join(frontend,'service/server.py');
  // UI assets are extraResources outside ASAR so Python can serve them directly.
  const servingBuild=app.isPackaged?path.join(process.resourcesPath,'ui'):path.join(frontend,'build');
  if(!fs.existsSync(python))throw new Error(`Python not found: ${python}\nSet NYS_FRONTEND_PYTHON or use the existing repository environment.`);
  if(!fs.existsSync(path.join(servingBuild,'index.html')))throw new Error('UI is not built. Run .\\frontend\\scripts\\build.ps1 first.');
  log=fs.createWriteStream(path.join(frontend,'logs/desktop-service.log'),{flags:'a'});
  service=spawn(python,['-B',server,'--frontend',frontend,'--build',servingBuild,'--ready-file',ownedReady],{
    windowsHide:true,stdio:['ignore','pipe','pipe'],env:{...process.env,PYTHONDONTWRITEBYTECODE:'1',GDAL_PAM_ENABLED:'NO',TEMP:path.join(frontend,'temp'),TMP:path.join(frontend,'temp')}
  });
  service.stdout.pipe(log,{end:false});service.stderr.pipe(log,{end:false});
  let failure;
  service.on('error',error=>{failure=error;});
  for(let attempt=0;attempt<600;attempt++){
    if(failure)throw failure;
    if(service.exitCode!==null)throw new Error(`Local service exited (${service.exitCode}). See frontend/logs/desktop-service.log.`);
    if(fs.existsSync(ownedReady)){endpoint=JSON.parse(fs.readFileSync(ownedReady,'utf8'));break;}
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  if(!endpoint)throw new Error('Imagery inventory startup timed out. See frontend/logs/desktop-service.log.');
  const origin=`http://127.0.0.1:${endpoint.port}`;
  window=new BrowserWindow({title:'NYS Imagery Search',width:1440,height:960,minWidth:900,minHeight:640,backgroundColor:'#edf1eb',autoHideMenuBar:true,show:false,
    webPreferences:{nodeIntegration:false,contextIsolation:true,sandbox:true,webSecurity:true}});
  window.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  window.webContents.on('will-navigate',(event,url)=>{if(new URL(url).origin!==origin)event.preventDefault();});
  const initial=process.env.NYS_INITIAL_VIEW?`?view=${encodeURIComponent(process.env.NYS_INITIAL_VIEW)}`:'';
  await window.loadURL(`${origin}/${initial}#${endpoint.token}`);
  showWindow();launchLog('Desktop window shown');
  if(process.env.NYS_UI_SMOKE==='1'){
    await new Promise(resolve=>setTimeout(resolve,3000));
    const snapshot=await window.webContents.executeJavaScript('({title:document.title, map:!!document.querySelector(".leaflet-container"), status:document.getElementById("coverage-text").textContent, tilesLoaded:document.querySelectorAll(".leaflet-tile-loaded").length})');
    snapshot.packaged=app.isPackaged;
    fs.writeFileSync(path.join(frontend,app.isPackaged?'logs/packaged-smoke.json':'logs/desktop-smoke.json'),JSON.stringify(snapshot,null,2));
    app.quit();
  }
}

if(!app.requestSingleInstanceLock())app.quit();
else{
  app.on('second-instance',showWindow);
  app.whenReady().then(start).catch(async error=>{launchLog(`Startup failed: ${error.stack||error.message}`);if(process.env.NYS_UI_SMOKE==='1')fs.writeFileSync(path.join(frontend,'logs/desktop-smoke-error.json'),JSON.stringify({error:error.message}));else dialog.showErrorBox('NYS Imagery Search',error.message);await stopService();closing=true;app.quit();});
  app.on('window-all-closed',()=>app.quit());
  app.on('before-quit',event=>{if(closing)return;event.preventDefault();closing=true;stopService().finally(()=>app.quit());});
}
