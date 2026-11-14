// Fresh-checkout browser smoke: generated imagery, real renderer, mocked text/model.
// Requires a built UI, development Python dependencies, Node 22+, and Edge.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';

const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repository = path.dirname(frontend);
const sessionRoot = path.join(frontend, 'temp', `synthetic-smoke-${Date.now()}`);
const job = path.join(sessionRoot, 'job');
const state = path.join(sessionRoot, 'state');
const profile = path.join(sessionRoot, 'edge-profile');
const artifact = path.join(frontend, 'tests/artifacts');
const python = process.env.FRONTEND_DEV_PYTHON || path.join(repository, '.venv-dev/Scripts/python.exe');
const edgePath = process.env.FRONTEND_EDGE_PATH || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const env = {...process.env, PYTHONDONTWRITEBYTECODE: '1', GDAL_PAM_ENABLED: 'NO'};
for (const directory of [profile, artifact]) await mkdir(directory, {recursive: true});

function child(script, args) {
  return spawn(python, ['-B', script, ...args], {windowsHide: true, env, stdio: ['ignore', 'ignore', 'pipe']});
}
async function finished(process, label) {
  let log = '';
  process.stderr.on('data', chunk => { log += chunk; });
  const code = await new Promise((resolve, reject) => { process.once('exit', resolve); process.once('error', reject); });
  assert.equal(code, 0, `${label} failed: ${log}`);
}
let service, edge, socket, session, origin;
let requestId = 0;
const pending = new Map();
const errors = [];
const requests = [];
const searchRequests = [];
async function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.onopen = () => resolve(ws);
    ws.onerror = reject;
  });
}
function command(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++requestId;
    const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${method}`)); }, 20000);
    pending.set(id, {resolve, reject, timeout});
    socket.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const result = await command('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  assert.equal(result.exceptionDetails, undefined, JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function waitFor(expression) {
  for (let attempt = 0; attempt < 300; attempt++) {
    if (await evaluate(expression)) return;
    await delay(100);
  }
  throw new Error(`Condition timed out: ${expression}`);
}
async function api(route, data) {
  return fetch(origin + route, {method: data === undefined ? 'GET' : 'POST',
    headers: {Authorization: `Bearer ${session.token}`, ...(data === undefined ? {} : {'Content-Type': 'application/json'})},
    body: data === undefined ? undefined : JSON.stringify(data)});
}

try {
  await finished(child(path.join(repository, 'scripts/create_synthetic_job.py'), ['--output', job]), 'Generator');
  service = child(path.join(frontend, 'service/dev_server.py'), ['--job', job, '--state', state]);
  let serviceLog = '';
  service.stderr.on('data', chunk => { serviceLog += chunk; });
  service.on('error', error => { serviceLog += error.message; });
  for (let attempt = 0; attempt < 300; attempt++) {
    if (service.exitCode !== null) throw new Error(`Synthetic service exited: ${serviceLog}`);
    try { session = JSON.parse(await readFile(path.join(state, 'ready.json'), 'utf8')); break; } catch {}
    await delay(100);
  }
  assert.ok(session?.synthetic, 'Synthetic service did not become ready');
  origin = `http://127.0.0.1:${session.port}`;
  const config = await (await api('/api/config')).json();
  assert.equal(config.synthetic, true);
  assert.equal((await fetch(origin + '/api/status')).status, 403);
  const [west, south, east, north] = config.imagery.bounds;
  const lon = (west + east) / 2, lat = (south + north) / 2;
  edge = spawn(edgePath, ['--headless=new', '--disable-gpu', '--remote-debugging-port=0',
    `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
    '--disable-background-networking', '--disable-component-update', '--disable-sync', '--disable-extensions', 'about:blank'],
    {windowsHide: true, env, stdio: 'ignore'});
  let launchError;
  edge.on('error', error => { launchError = error; });
  let port;
  for (let attempt = 0; attempt < 200; attempt++) {
    if (launchError) throw launchError;
    try { port = Number((await readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]); if (port) break; } catch {}
    await delay(100);
  }
  assert.ok(port, 'Edge debugging endpoint unavailable');
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket = await connect(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    const task = pending.get(message.id);
    if (task) { pending.delete(message.id); clearTimeout(task.timeout); message.error ? task.reject(new Error(message.error.message)) : task.resolve(message.result); }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
    if (message.method === 'Network.requestWillBeSent') {
      const request = message.params.request;
      requests.push(request.url);
      if (request.method === 'POST' && request.url === origin + '/api/search' && request.postData) {
        searchRequests.push(JSON.parse(request.postData));
      }
    }
  };
  await command('Runtime.enable');
  await command('Page.enable');
  await command('Network.enable');
  await command('Emulation.setDeviceMetricsOverride', {width: 1440, height: 960, deviceScaleFactor: 1, mobile: false});
  await command('Page.navigate', {url: `${origin}/?view=${lon},${lat},18#${session.token}`});
  await waitFor("document.getElementById('coverage-text')?.textContent.includes('2 of 2') && document.querySelector('.leaflet-tile-loaded')");
  await waitFor("document.getElementById('synthetic-banner') && !document.getElementById('synthetic-banner').classList.contains('hidden')");
  assert.equal(await evaluate("document.getElementById('top-k').disabled && document.getElementById('fixed-result-count').classList.contains('hidden')"), true);
  await evaluate("document.getElementById('query').value='synthetic patterns'; document.getElementById('top-k').value='3'; document.getElementById('search-form').requestSubmit()");
  await waitFor("document.querySelectorAll('.result-card').length===6");
  assert.equal(searchRequests.at(-1).detection.enabled, true);
  assert.ok(!Object.hasOwn(searchRequests.at(-1), 'top_k'));
  assert.match(await evaluate("document.querySelector('.result-card .box-confidence')?.textContent"), /0\.920 box confidence/);
  assert.match(await evaluate("document.getElementById('result-order').textContent"), /highest box confidence/);
  await evaluate("document.querySelector('.result-card').click()");
  await waitFor("document.querySelector('.selected-preview')?.complete && document.querySelector('.selected-preview')?.naturalWidth===256");
  await waitFor("document.querySelector('.detection-boxes rect.touches-tile-edge')?.getAttribute('width')==='140' && document.querySelector('.leaflet-object-boxes-pane path')?.getAttribute('stroke-dasharray')==='6 4'");
  assert.match(await evaluate("document.querySelector('.detection-detail').textContent"), /Mock detections.*Mock object/);
  assert.match(await evaluate("document.querySelector('.edge-note').textContent"), /May extend beyond this tile/);
  assert.match(await evaluate("document.querySelector('.detection-detail').textContent"), /1 large partial box hidden/);
  assert.equal(await evaluate("document.querySelectorAll('.detection-boxes rect').length"), 1);
  assert.equal(await evaluate("document.querySelectorAll('.leaflet-object-boxes-pane path').length"), 1);
  await evaluate("document.getElementById('partial-box-toggle').click()");
  await waitFor("document.getElementById('partial-box-toggle').getAttribute('aria-pressed')==='true' && document.querySelector('.detection-boxes rect.touches-tile-edge[width=\"200\"]') && document.querySelectorAll('.leaflet-object-boxes-pane path[stroke-dasharray=\"6 4\"]').length===2");
  assert.match(await evaluate("document.getElementById('search-scope').textContent"), /Mock detections: 6.*7 tiles checked.*no more candidates/);
  assert.equal(await evaluate("document.querySelectorAll('path.detection-tile-outline').length"), 0);
  await evaluate("document.getElementById('tile-outlines').click()");
  await waitFor("document.querySelector('path.detection-tile-outline') && document.getElementById('tile-outlines').getAttribute('aria-pressed')==='true'");
  // Older saved results used similarity order and did not mark edge-clipped boxes.
  // Simulate that format in this run's isolated synthetic state.
  const [history] = await (await api('/api/history')).json();
  const legacyPath = path.join(state, 'state/searches', history.id + '.json');
  const legacy = JSON.parse(await readFile(legacyPath, 'utf8'));
  legacy.results.sort((a, b) => b.score - a.score);
  assert.notEqual(legacy.results[0].detection.maximum_box_score, 0.92);
  for (const [index, row] of legacy.results.entries()) {
    row.rank = index + 1;
    for (const box of row.detection.predictions) delete box.touches_tile_edge;
  }
  await writeFile(legacyPath, JSON.stringify(legacy));
  await evaluate("document.getElementById('history').click()");
  await waitFor("document.querySelector('[data-history]')");
  await evaluate("document.querySelector('[data-history]').click()");
  await waitFor("!document.getElementById('dialog').open && document.querySelectorAll('.result-card').length===6 && document.querySelector('.detection-boxes rect')");
  assert.match(await evaluate("document.querySelector('.result-card .box-confidence')?.textContent"), /0\.920 box confidence/);
  await waitFor("document.querySelector('.detection-boxes rect.touches-tile-edge') && !document.querySelector('path.detection-tile-outline')");
  assert.equal(await evaluate("document.getElementById('partial-box-toggle').getAttribute('aria-pressed')"), 'false');
  assert.equal(await evaluate("document.querySelectorAll('.detection-boxes rect').length"), 1);
  await evaluate("document.getElementById('load-model').click()");
  await waitFor("!document.getElementById('unload-model').classList.contains('hidden')");
  const model = await (await api('/api/status')).json();
  assert.equal(model.model.synthetic, true);
  assert.match(model.model.gpu, /MOCK/);
  assert.equal(await evaluate("!document.getElementById('synthetic-banner').classList.contains('hidden') && document.querySelector('.leaflet-control-attribution').textContent.includes('SYNTHETIC')"), true);
  const screenshot = await command('Page.captureScreenshot', {format: 'png'});
  await writeFile(path.join(artifact, 'synthetic-ui.png'), Buffer.from(screenshot.data, 'base64'));
  await evaluate("document.getElementById('settings').click()");
  await waitFor("document.getElementById('box-threshold')");
  await evaluate("const threshold=document.getElementById('box-threshold'); threshold.value='1'; threshold.dispatchEvent(new Event('change')); document.getElementById('close-dialog').click(); document.getElementById('search-form').requestSubmit()");
  await waitFor("document.getElementById('search-scope').textContent.includes('Mock detections: 0') && !document.querySelector('.selected-preview') && !document.querySelector('.leaflet-object-boxes-pane path')");
  await evaluate("document.getElementById('detect-objects').click(); document.getElementById('search-form').requestSubmit()");
  await waitFor("document.querySelectorAll('.result-card').length===3 && !document.querySelector('.detection-boxes')");
  assert.equal(await evaluate("!document.getElementById('top-k').disabled && !document.getElementById('fixed-result-count').classList.contains('hidden')"), true);
  assert.equal(searchRequests.at(-1).detection.enabled, false);
  assert.equal(searchRequests.at(-1).top_k, 3);
  assert.equal(errors.length, 0, JSON.stringify(errors));
  assert.ok(requests.some(url => url.includes('/api/tiles/')), 'Real display tiles must be requested');
  const external = requests.filter(url => !(url.startsWith(origin + '/') || url.startsWith('blob:' + origin + '/') || url === 'about:blank' || url.startsWith('data:')));
  assert.equal(external.length, 0, `UI must remain offline: ${external.map(url => { const parsed = new URL(url); return parsed.origin + parsed.pathname; }).join(', ')}`);
  await writeFile(path.join(artifact, 'synthetic-report.json'), JSON.stringify({passed: true, synthetic: true,
    sourceTiles: 2, vectorRows: 7, detectionResults: 6, similarityResults: 3, runtimeErrors: errors.length}, null, 2));
  console.log('Synthetic UI smoke passed: real tiles/previews, mock detection refill and boxes, saved history, auth, no external requests.');
} finally {
  socket?.close();
  for (const task of pending.values()) clearTimeout(task.timeout);
  if (origin) { try { await api('/api/shutdown', {}); } catch {} }
  if (service && service.exitCode === null) {
    for (let attempt = 0; attempt < 30 && service.exitCode === null; attempt++) await delay(100);
    if (service.exitCode === null) service.kill();
  }
  if (edge && edge.exitCode === null) edge.kill();
}
