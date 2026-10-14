"""Authenticated HTTP transport for the local application.

Keep process ownership and application behavior in server.Application. This module
owns request validation, route dispatch, bounded connections, and response headers.
It does not import or construct Application, so tests can supply an isolated fake.
"""

import json
import mimetypes
import re
import secrets
import threading
from concurrent.futures import CancelledError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from storage import contained


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.application = app
        self.connections = threading.BoundedSemaphore(16)
        super().__init__(address, Handler)

    def process_request(self, request, address):
        if not self.connections.acquire(blocking=False):
            request.sendall(b"HTTP/1.0 503 Busy\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            self.shutdown_request(request)
            return
        super().process_request(request, address)

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.connections.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def log_message(self, format, *args):
        pass

    def send(self, status, blob, mime="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(blob)))
        self.send_header(
            "Cache-Control", "private, max-age=86400" if mime == "image/png" else "no-store"
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; object-src 'none'",
        )
        try:
            self.end_headers()
            self.wfile.write(blob)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def json(self, value, status=200):
        self.send(status, json.dumps(value, allow_nan=False).encode("utf-8"))

    def validate_access(self, api, query):
        expected = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != expected:
            raise PermissionError("Invalid loopback host")
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{expected}":
            raise PermissionError("Cross-origin access is disabled")
        if api:
            supplied = (
                self.headers.get("Authorization", "").removeprefix("Bearer ")
                or query.get("token", [""])[0]
            )
            if not secrets.compare_digest(supplied, self.server.application.token):
                raise PermissionError("Invalid session token")

    def do_GET(self):
        app = self.server.application
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        route = unquote(url.path)
        try:
            self.validate_access(route.startswith("/api/"), query)
            base_only = query.get("base_only", ["0"])[0] == "1"
            if route == "/api/config":
                return self.json(app.config(base_only))
            if route == "/api/status":
                return self.json(app.status(base_only))
            if route == "/api/history":
                return self.json(app.history())
            if route == "/api/bookmarks":
                return self.json(app.bookmarks())
            if route == "/api/results":
                return self.json(app.saved(query.get("id", [""])[0]))
            if route == "/api/search-progress":
                return self.json(app.search_progress(query.get("id", [""])[0]))
            if route == "/api/inspect":
                return self.json(
                    app.map_work(
                        (app.base_renderer if base_only else app.renderer).inspect,
                        float(query["lon"][0]),
                        float(query["lat"][0]),
                    )
                )
            if route == "/api/preview":
                uri = query.get("source", [""])[0]
                source = app.inventory.by_uri.get(uri)
                if not source:
                    raise ValueError("Unknown source")
                row = {
                    "source_uri": uri,
                    "source_sha256": source.sha256,
                    "pixel_x": int(query["x"][0]),
                    "pixel_y": int(query["y"][0]),
                    "pixel_width": 256,
                    "pixel_height": 256,
                }
                return self.send(
                    200,
                    app.map_work(app.renderer.preview, row, query.get("context", ["0"])[0] == "1"),
                    "image/png",
                )
            match = re.fullmatch("/api/tiles/(\\d+)/(\\d+)/(\\d+)\\.png", route)
            if match:
                return self.send(
                    200,
                    app.map_work(
                        (app.base_renderer if base_only else app.renderer).tile,
                        *map(int, match.groups()),
                        connection=self.connection,
                        tile=True,
                    ),
                    "image/png",
                )
            if route.startswith("/api/"):
                return self.json({"error": "Unknown API endpoint"}, 404)
            static = (
                app.build / "index.html"
                if route == "/"
                else contained(app.build, app.build / route.lstrip("/"))
            )
            if not static.is_file():
                return self.json({"error": "File not found"}, 404)
            mime = mimetypes.guess_type(static.name)[0] or "application/octet-stream"
            if static.suffix == ".js":
                mime = "text/javascript"
            self.send(200, static.read_bytes(), mime)
        except CancelledError:
            pass
        except PermissionError as exc:
            self.json({"error": str(exc)}, 403)
        except (ValueError, KeyError, FileNotFoundError, AttributeError) as exc:
            self.json({"error": str(exc)}, 400)
        except RuntimeError as exc:
            self.json({"error": str(exc)}, 503)
        except Exception as exc:
            self.json({"error": str(exc)}, 500)

    def do_POST(self):
        app = self.server.application
        url = urlsplit(self.path)
        try:
            self.validate_access(True, parse_qs(url.query))
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise ValueError("Invalid request size")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Expected JSON")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("Expected an object")
            if url.path == "/api/search":
                return self.json(app.start_search(value), 202)
            if url.path == "/api/model/load":
                return self.json(app.start_model(value), 202)
            if url.path == "/api/model/unload":
                return self.json(app.stop_model())
            if url.path == "/api/overview/prepare":
                if value.get("confirm") != "prepare-local-overview":
                    raise ValueError("Confirm prepare-local-overview")
                return self.json(app.start_overview(value.get("base_only", False)), 202)
            if url.path == "/api/bookmarks":
                return self.json(app.save_bookmark(value))
            if url.path == "/api/history/clear":
                confirmation = value.get("confirm")
                if confirmation not in ("frontend-history-only", "all-search-history"):
                    raise ValueError("Confirm all-search-history or frontend-history-only")
                return self.json(
                    app.clear_history(include_backend=confirmation == "all-search-history")
                )
            if url.path == "/api/cache/clear":
                if value.get("confirm") != "frontend-cache-only":
                    raise ValueError("Confirm frontend-cache-only")
                if app.overview_thread and app.overview_thread.is_alive():
                    raise ValueError("Wait for overview preparation before clearing the cache")
                app.cache.clear()
                return self.json(app.cache.stats())
            if url.path == "/api/shutdown":
                self.json({"state": "closing"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self.json({"error": "Unknown endpoint"}, 404)
        except PermissionError as exc:
            self.json({"error": str(exc)}, 403)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            self.json({"error": str(exc)}, 400)
        except RuntimeError as exc:
            self.json({"error": str(exc)}, 409)
        except Exception as exc:
            self.json({"error": str(exc)}, 500)
