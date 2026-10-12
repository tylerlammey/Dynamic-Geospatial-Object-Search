"""Authenticated loopback service, bounded map work, serialized owned search."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor,TimeoutError as FutureTimeout,CancelledError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import signal
import select
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

from imagery import Inventory, TileRenderer
from retrieval import BUSY_MESSAGE, status_snapshot, validate_query, validate_search_options
from storage import DiskCache, atomic_json, contained, frontend_ownership, retry_windows_file_operation
from overview import prepare_overview


class Application:
    def __init__(self,frontend,build):
        self.frontend=Path(frontend).resolve();self.repository=self.frontend.parent
        self.build=Path(build).resolve();self.job=self.repository/'data/nys-naip-2022-rgb-060m'
        self.collection=self.job/'embeddings/georsclip-vith14-s256-st192'
        sys.path.insert(0,str(self.repository/'src'))
        self.settings=json.loads((self.frontend/'settings.json').read_text(encoding='utf-8'))
        limit=self.settings['cache_limit_bytes'];workers=self.settings['imagery_workers']
        if type(limit) is not int or not 1_000_000<=limit<=20_000_000_000:raise ValueError('Cache limit must be 1 MB to 20 GB')
        if type(workers) is not int or not 1<=workers<=4:raise ValueError('Use 1 to 4 imagery workers')
        self.cache=DiskCache(self.frontend/'cache',limit)
        self.inventory=None;self.renderer=None;self.imagery_error=None
        try:
            self.inventory=Inventory(self.job)
            self.renderer=TileRenderer(self.inventory,self.cache,workers)
        except (FileNotFoundError,ValueError) as exc:self.imagery_error=str(exc)
        self.token=secrets.token_urlsafe(32)
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='imagery')
        self.map_slots=threading.BoundedSemaphore(workers+12)
        self.search_lock=threading.Lock();self.search=None;self.search_id=None;self.search_log=None
        self.state_lock=threading.RLock();self.poll_lock=threading.Lock();self.last_status=None;self.status_at=0
        self.overview_lock=threading.Lock();self.overview_thread=None;self.overview_cancel=threading.Event()
        self.overview_status={'state':'complete' if self.renderer and self.renderer.overview_ready() else 'idle','ready':bool(self.renderer and self.renderer.overview_ready())}
        for name in ('logs','state','state/searches','temp'):(self.frontend/name).mkdir(parents=True,exist_ok=True)

    def status(self):
        with self.poll_lock:
            if self.last_status is None or time.monotonic()-self.status_at>3:
                try:self.last_status=status_snapshot(self.collection,self.job)
                except Exception as exc:self.last_status={'state':'error','error':str(exc),'busy':True,'published_sources':0,'published_chips':0,'total_sources':3805,'coverage':[],'partial':True}
                self.status_at=time.monotonic()
            return {**self.last_status,'cache':self.cache.stats(),'imagery_error':self.imagery_error,'metrics':self.renderer.metrics if self.renderer else {},
                    'overview':{**self.overview_status,'ready':bool(self.renderer and self.renderer.overview_ready()),'epoch':self.renderer.display_epoch if self.renderer else 0}}

    def config(self):
        sources=[] if not self.inventory else [{'uri':s.uri,'bounds':s.bounds} for s in self.inventory.sources]
        return {'imagery':self.inventory.public() if self.inventory else None,'imagery_error':self.imagery_error,'sources':sources,
                'tile_size':256,'native_resolution':.6,'attribution':'Courtesy USDA-FSA; distributed by NOAA Digital Coast'}

    def map_work(self,operation,*args,connection=None,tile=False):
        if not self.renderer:raise ValueError(self.imagery_error or 'Imagery unavailable')
        if not self.map_slots.acquire(blocking=False):raise RuntimeError('Imagery queue is full; retry shortly')
        cancel=threading.Event()
        future=self.pool.submit(operation,*args,**({'cancel':cancel} if tile else {}))
        try:
            while True:
                try:return future.result(timeout=.2)
                except FutureTimeout:
                    if connection:
                        try:
                            readable,_,_=select.select([connection],[],[],0)
                            if readable and connection.recv(1,socket.MSG_PEEK)==b'':
                                cancel.set();future.cancel();raise CancelledError('Obsolete browser tile request')
                        except (OSError,ValueError):
                            cancel.set();future.cancel();raise CancelledError('Browser disconnected')
        finally:self.map_slots.release()

    def start_overview(self):
        if not self.renderer:raise ValueError('Downloaded imagery is unavailable')
        self.status_at=0;status=self.status()
        if status['busy'] and status['state'] in ('running','pausing'):raise RuntimeError('Embedding is running. Prepare the display overview while embeddings are idle or manually paused.')
        with self.overview_lock:
            if self.overview_thread and self.overview_thread.is_alive():return self.overview_status
            self.overview_cancel.clear()
            workers=min(8,os.cpu_count() or 2)
            self.overview_status={'state':'starting','ready':self.renderer.overview_ready(),'total_sources':len(self.inventory.sources),'completed_sources':0,'workers':workers}
            last_log=[0.]
            def progress(value):
                self.overview_status={**value,'workers':workers}
                now=time.monotonic()
                if value['state']!='preparing' or now-last_log[0]>=1:
                    atomic_json(self.frontend/'logs/overview-status.json',self.overview_status,self.frontend);last_log[0]=now
            def embedding_active():
                current=self.status()
                return current['busy'] and current['state'] in ('running','pausing')
            def build():
                if os.name=='nt':
                    import ctypes
                    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
                try:prepare_overview(self.renderer,progress,workers,self.overview_cancel,embedding_active)
                except CancelledError as exc:progress({'state':'cancelled','error':str(exc)})
                except Exception as exc:progress({'state':'error','error':str(exc)})
                finally:
                    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            self.overview_thread=threading.Thread(target=build,name='overview-preparation',daemon=True);self.overview_thread.start()
            return self.overview_status

    def history(self):
        with self.state_lock:
            records=[]
            for p in sorted((self.frontend/'state/searches').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:30]:
                try:
                    data=json.loads(p.read_text(encoding='utf-8'));records.append({'id':p.stem,'query':data['query'],'count':len(data['results']),'searched_source_tiles':data['searched_source_tiles'],'total_source_tiles':data['total_source_tiles'],'created_utc':data['created_utc']})
                except (ValueError,KeyError):continue
            records.extend({k:v for k,v in r.items() if k!='backend_path'} for r in self.backend_history())
            return sorted(records,key=lambda r:r['created_utc'],reverse=True)[:50]

    def clear_history(self):
        """Remove disposable frontend searches only, while no query can publish."""
        with self.search_lock, self.state_lock:
            if self.search and self.search.poll() is None:
                raise RuntimeError('Wait for the running search to finish before clearing history')
            files=[];folders=[]
            for relative,pattern in (
                ('state/searches',r'[a-f0-9]{32}\.json(?:\.[a-f0-9]{32}\.tmp)?'),
                ('state/text-vectors',r'[a-f0-9]{64}\.json(?:\.[a-f0-9]{32}\.tmp)?'),
                ('logs/search',None),
            ):
                directory=self.frontend/relative
                if directory.is_symlink() or contained(self.frontend,directory)!=directory.absolute():
                    raise ValueError('History directories cannot be redirected')
                if not directory.exists():continue
                for path in directory.iterdir():
                    if pattern:
                        if re.fullmatch(pattern,path.name):files.append(path)
                    elif re.fullmatch('[a-f0-9]{32}',path.name):
                        if path.is_symlink() or contained(directory,path)!=path.absolute() or not path.is_dir():
                            raise ValueError('Search log folders cannot be redirected')
                        folders.append(path)
                        for child in path.iterdir():
                            if not re.fullmatch(r'(?:worker\.log|(?:request|status)\.json(?:\.[a-f0-9]{32}\.tmp)?)',child.name):
                                raise ValueError('Unexpected file in search logs; cleanup cancelled')
                            files.append(child)
            # Validate every target before deleting anything. Never follow links
            # or recurse into an unexpected directory, including Windows junctions.
            for path in files:
                if path.is_symlink() or contained(self.frontend,path)!=path.absolute() or not path.is_file():
                    raise ValueError('History files must be regular frontend files')
            if self.search_log:
                self.search_log.close();self.search_log=None
            freed=0
            for path in files:
                size=path.stat().st_size
                retry_windows_file_operation(path.unlink);freed+=size
            for folder in folders:retry_windows_file_operation(folder.rmdir)
            self.search=None;self.search_id=None
            return {'removed_files':len(files),'freed_bytes':freed}

    def backend_history(self):
        """Expose compatible existing CLI searches without changing their files."""
        records=[];fingerprint=self.status().get('run_fingerprint')
        for p in sorted((self.job/'search-results').glob('*/results.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:50]:
            try:
                data=json.loads(p.read_text(encoding='utf-8'))
                if data.get('run_fingerprint')!=fingerprint or Path(data.get('collection_dir','')).resolve()!=self.collection.resolve():continue
                identifier='backend-'+hashlib.sha256(p.relative_to(self.job).as_posix().encode()).hexdigest()[:32]
                records.append({'id':identifier,'query':data['query'],'count':len(data['results']),
                                'searched_source_tiles':data['searched_source_tiles'],'total_source_tiles':data['total_source_tiles'],
                                'created_utc':p.stat().st_mtime,'backend_path':p,'cached_backend':True})
            except (ValueError,KeyError):continue
        return records

    def saved(self,identifier):
        if re.fullmatch('backend-[a-f0-9]{32}',identifier):return self.saved_backend(identifier)
        if not re.fullmatch('[a-f0-9]{32}',identifier):raise ValueError('Invalid result ID')
        path=contained(self.frontend,self.frontend/'state/searches'/f'{identifier}.json')
        result=json.loads(path.read_text(encoding='utf-8'));result['result_id']=identifier
        status=self.status()
        if status.get('run_fingerprint') and result.get('run_fingerprint')!=status['run_fingerprint']:raise ValueError('Saved results belong to another collection fingerprint')
        return result

    def saved_backend(self,identifier):
        record=next((r for r in self.backend_history() if r['id']==identifier),None)
        if record is None:raise ValueError('Compatible saved search was not found')
        data=json.loads(record['backend_path'].read_text(encoding='utf-8'))
        # Published bundles are immutable. Read only matching metadata groups;
        # no model, GPU lock, pause signal, or backend output write is involved.
        from geospatial_search.bundles import selected_rows,validate_bundle
        from shapely import from_wkb
        catalog=json.loads((self.collection/'collection.json').read_text(encoding='utf-8'))
        if catalog['run_fingerprint']!=data['run_fingerprint']:raise ValueError('Saved search fingerprint differs')
        published={s['shard_id']:s for s in catalog['shards']};groups={}
        for result in data['results']:
            if result['shard_id'] not in published:raise ValueError('Saved result is not published')
            groups.setdefault(result['shard_id'],[]).append(result['embedding_id'])
        metadata={}
        for shard,ids in groups.items():
            folder=contained(self.collection/'shards',self.collection/'shards'/shard)
            run=validate_bundle(folder,data['run_fingerprint'])
            entry=published[shard]
            if run['source_uri']!=entry['source_uri'] or run['chip_count']!=entry['chip_count']:raise ValueError('Saved bundle identity differs from catalog')
            metadata[shard]=selected_rows(folder/'chips.geoparquet',ids)
        results=[]
        for saved in data['results']:
            row=metadata[saved['shard_id']][saved['embedding_id']]
            for field in ('source_uri','source_sha256','pixel_x','pixel_y','pixel_width','pixel_height','chip_id'):
                if row[field]!=saved[field]:raise ValueError('Saved result identity differs from original metadata')
            results.append({**saved,'geometry':from_wkb(row['geometry']).__geo_interface__})
        return {**{key:value for key,value in data.items() if key not in ('results','collection_dir')},
                'results':results,'created_utc':record['created_utc'],'result_id':identifier,'cached_backend':True}

    def start_search(self,value):
        query,top_k,allow_partial=validate_query(value)
        backend,effort,seen=validate_search_options(value)
        with self.search_lock:
            if self.search and self.search.poll() is None:raise RuntimeError('A frontend search is already running')
            # Refresh the read-only status before creating any query process.
            self.status_at=0;status=self.status()
            if status.get('error'):raise ValueError(status['error'])
            if status['busy']:raise RuntimeError(BUSY_MESSAGE)
            if not status['published_chips']:raise ValueError('No saved embeddings yet; imagery browsing remains available')
            if status['collection_state'] not in ('complete','paused') and not allow_partial:raise ValueError('Interrupted collection: explicitly enable searching an idle incomplete collection in Settings')
            identifier=uuid.uuid4().hex;folder=self.frontend/'logs/search'/identifier
            request={'task_id':identifier,'query':query,'top_k':top_k,'allow_partial':allow_partial,
                     'offline_model':self.settings.get('offline_model',True),'search_backend':backend,'search_effort':effort,'exclude_ids':seen}
            atomic_json(folder/'request.json',request,self.frontend)
            atomic_json(folder/'status.json',{'state':'starting','task_id':identifier},self.frontend)
            if self.search_log:self.search_log.close()
            self.search_log=(folder/'worker.log').open('wb')
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','GDAL_PAM_ENABLED':'NO','TEMP':str(self.frontend/'temp'),'TMP':str(self.frontend/'temp')}
            worker=Path(__file__).with_name('search_worker.py')
            self.search=subprocess.Popen([sys.executable,'-B',str(worker),'--frontend',str(self.frontend),'--request',str(folder/'request.json')],stdout=self.search_log,stderr=self.search_log,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.search_id=identifier
            return {'task_id':identifier}

    def search_progress(self,identifier):
        if not re.fullmatch('[a-f0-9]{32}',identifier):raise ValueError('Invalid task ID')
        data=json.loads(contained(self.frontend,self.frontend/'logs/search'/identifier/'status.json').read_text(encoding='utf-8'))
        if identifier==self.search_id and self.search and self.search.poll() is None and data['state']=='complete':
            # Results may be published just before native process teardown.
            # Publish completion only when a following serialized query can start.
            return {**data,'state':'finishing'}
        if identifier==self.search_id and self.search and self.search.poll() is not None and data['state'] not in ('complete','error'):
            data={'state':'error','error':'Query process stopped. See the frontend query log.'}
        return data

    def bookmarks(self):
        p=self.frontend/'state/bookmarks.json'
        try:return json.loads(p.read_text(encoding='utf-8'))
        except FileNotFoundError:return []

    def save_bookmark(self,row):
        if not self.inventory or row.get('source_uri') not in self.inventory.by_uri:raise ValueError('Invalid bookmark source')
        # Validate result windows and geometry before persisting arbitrary client data.
        for name in ('pixel_x','pixel_y','pixel_width','pixel_height'):
            if type(row.get(name)) is not int:raise ValueError('Invalid bookmark pixel window')
        source=self.inventory.by_uri[row['source_uri']]
        if row['pixel_width']!=256 or row['pixel_height']!=256 or row['pixel_x']<0 or row['pixel_y']<0 or row['pixel_x']+256>source.width or row['pixel_y']+256>source.height:raise ValueError('Invalid bookmark window')
        if not isinstance(row.get('chip_id'),str):raise ValueError('Missing chip identity')
        with self.state_lock:
            records=self.bookmarks();key=row['chip_id']
            if any(r['chip_id']==key for r in records):records=[r for r in records if r['chip_id']!=key]
            else:records=([row]+records)[:500]
            atomic_json(self.frontend/'state/bookmarks.json',records,self.frontend)
            return records

    def shutdown(self):
        self.overview_cancel.set()
        if self.renderer:self.renderer.close()
        # Only the child stored by this frontend is eligible for termination.
        if self.search and self.search.poll() is None:
            self.search.terminate()
            try:self.search.wait(timeout=5)
            except subprocess.TimeoutExpired:self.search.kill()
        if self.search_log:self.search_log.close()
        self.pool.shutdown(wait=False,cancel_futures=True)


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,app):
        self.application=app;self.connections=threading.BoundedSemaphore(16)
        super().__init__(address,Handler)
    def process_request(self,request,address):
        if not self.connections.acquire(blocking=False):
            request.sendall(b'HTTP/1.0 503 Busy\r\nConnection: close\r\nContent-Length: 0\r\n\r\n');self.shutdown_request(request);return
        super().process_request(request,address)
    def process_request_thread(self,request,address):
        try:super().process_request_thread(request,address)
        finally:self.connections.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def setup(self):
        super().setup();self.connection.settimeout(15)
    def handle(self):
        try:super().handle()
        except (ConnectionResetError,ConnectionAbortedError,BrokenPipeError):pass
    def log_message(self,format,*args):pass
    def send(self,status,blob,mime='application/json'):
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(blob)))
        self.send_header('Cache-Control','private, max-age=86400' if mime=='image/png' else 'no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; object-src 'none'")
        try:self.end_headers();self.wfile.write(blob)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
    def json(self,value,status=200):self.send(status,json.dumps(value,allow_nan=False).encode('utf-8'))
    def validate_access(self,api,query):
        expected=f'127.0.0.1:{self.server.server_port}'
        if self.headers.get('Host')!=expected:raise PermissionError('Invalid loopback host')
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://{expected}':raise PermissionError('Cross-origin access is disabled')
        if api:
            supplied=self.headers.get('Authorization','').removeprefix('Bearer ') or query.get('token',[''])[0]
            if not secrets.compare_digest(supplied,self.server.application.token):raise PermissionError('Invalid session token')
    def do_GET(self):
        app=self.server.application;url=urlsplit(self.path);query=parse_qs(url.query);route=unquote(url.path)
        try:
            self.validate_access(route.startswith('/api/'),query)
            if route=='/api/config':return self.json(app.config())
            if route=='/api/status':return self.json(app.status())
            if route=='/api/history':return self.json(app.history())
            if route=='/api/bookmarks':return self.json(app.bookmarks())
            if route=='/api/results':return self.json(app.saved(query.get('id',[''])[0]))
            if route=='/api/search-progress':return self.json(app.search_progress(query.get('id',[''])[0]))
            if route=='/api/inspect':return self.json(app.map_work(app.renderer.inspect,float(query['lon'][0]),float(query['lat'][0])))
            if route=='/api/preview':
                uri=query.get('source',[''])[0];source=app.inventory.by_uri.get(uri)
                if not source:raise ValueError('Unknown source')
                row={'source_uri':uri,'source_sha256':source.sha256,'pixel_x':int(query['x'][0]),'pixel_y':int(query['y'][0]),'pixel_width':256,'pixel_height':256}
                return self.send(200,app.map_work(app.renderer.preview,row,query.get('context',['0'])[0]=='1'),'image/png')
            match=re.fullmatch(r'/api/tiles/(\d+)/(\d+)/(\d+)\.png',route)
            if match:return self.send(200,app.map_work(app.renderer.tile,*map(int,match.groups()),connection=self.connection,tile=True),'image/png')
            if route.startswith('/api/'):return self.json({'error':'Unknown API endpoint'},404)
            static=app.build/'index.html' if route=='/' else contained(app.build,app.build/route.lstrip('/'))
            if not static.is_file():return self.json({'error':'File not found'},404)
            mime=mimetypes.guess_type(static.name)[0] or 'application/octet-stream'
            if static.suffix=='.js':mime='text/javascript'
            self.send(200,static.read_bytes(),mime)
        except CancelledError:pass
        except PermissionError as exc:self.json({'error':str(exc)},403)
        except (ValueError,KeyError,FileNotFoundError,AttributeError) as exc:self.json({'error':str(exc)},400)
        except RuntimeError as exc:self.json({'error':str(exc)},503)
        except Exception as exc:self.json({'error':str(exc)},500)
    def do_POST(self):
        app=self.server.application;url=urlsplit(self.path)
        try:
            self.validate_access(True,parse_qs(url.query))
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=32768:raise ValueError('Invalid request size')
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Expected JSON')
            value=json.loads(self.rfile.read(length))
            if not isinstance(value,dict):raise ValueError('Expected an object')
            if url.path=='/api/search':return self.json(app.start_search(value),202)
            if url.path=='/api/overview/prepare':
                if value.get('confirm')!='prepare-local-overview':raise ValueError('Confirm prepare-local-overview')
                return self.json(app.start_overview(),202)
            if url.path=='/api/bookmarks':return self.json(app.save_bookmark(value))
            if url.path=='/api/history/clear':
                if value.get('confirm')!='frontend-history-only':raise ValueError('Confirm frontend-history-only')
                return self.json(app.clear_history())
            if url.path=='/api/cache/clear':
                if value.get('confirm')!='frontend-cache-only':raise ValueError('Confirm frontend-cache-only')
                if app.overview_thread and app.overview_thread.is_alive():raise ValueError('Wait for overview preparation before clearing the cache')
                app.cache.clear();return self.json(app.cache.stats())
            if url.path=='/api/shutdown':
                self.json({'state':'closing'});threading.Thread(target=self.server.shutdown,daemon=True).start();return
            self.json({'error':'Unknown endpoint'},404)
        except PermissionError as exc:self.json({'error':str(exc)},403)
        except (ValueError,KeyError,FileNotFoundError) as exc:self.json({'error':str(exc)},400)
        except RuntimeError as exc:self.json({'error':str(exc)},409)
        except Exception as exc:self.json({'error':str(exc)},500)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frontend',required=True);parser.add_argument('--build');parser.add_argument('--ready-file',required=True);parser.add_argument('--port',type=int,default=0)
    args=parser.parse_args();frontend=Path(args.frontend).resolve()
    with frontend_ownership(frontend):
        app=Application(frontend,args.build or frontend/'build')
        server=Server(('127.0.0.1',args.port),app)
        ready=contained(frontend,Path(args.ready_file))
        atomic_json(ready,{'port':server.server_port,'token':app.token,'pid':os.getpid()},frontend)
        try:server.serve_forever(poll_interval=.25)
        finally:server.server_close();app.shutdown();ready.unlink(missing_ok=True)


if __name__=='__main__':main()
