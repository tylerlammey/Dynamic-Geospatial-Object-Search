"""HTTP boundary regressions using an isolated app, no imagery or GPU processes."""

import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from http_server import Server


class HTTPBoundary(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parent / ".tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temporary.name)
        build = self.root / "build"
        build.mkdir()
        (build / "index.html").write_text("synthetic interface", encoding="utf-8")
        (self.root / "private.txt").write_text("private fixture", encoding="utf-8")
        self.app = SimpleNamespace(
            token="synthetic-session",
            build=build,
            status=Mock(return_value={"state": "paused"}),
            start_search=Mock(return_value={"task_id": "synthetic-task"}),
        )
        self.server = Server(("127.0.0.1", 0), self.app)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def tearDown(self):
        if self.thread.is_alive():
            self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def request(self, route, method="GET", body=None, authenticated=True, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        request_headers = {"Connection": "close"}
        if authenticated:
            request_headers["Authorization"] = "Bearer " + self.app.token
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        try:
            connection.request(method, route, body, request_headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_api_requires_session_and_accepts_query_token_access(self):
        self.assertEqual(self.request("/api/status", authenticated=False)[0], 403)
        self.assertEqual(self.request("/api/status?token=wrong", authenticated=False)[0], 403)
        self.assertEqual(
            self.request("/api/status?token=synthetic-session", authenticated=False)[0], 200
        )
        self.assertEqual(self.request("/api/status")[0], 200)
        self.assertEqual(self.app.status.call_count, 2)

    def test_host_and_origin_are_checked_before_dispatch_even_with_token(self):
        for headers in (
            {"Host": "localhost:" + str(self.server.server_port)},
            {"Host": "attacker.example"},
            {"Origin": "https://attacker.example"},
            {"Origin": "null"},
        ):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("/api/status", headers=headers)[0], 403)
        self.app.status.assert_not_called()
        origin = "http://127.0.0.1:" + str(self.server.server_port)
        self.assertEqual(
            self.request("/api/status?base_only=1", headers={"Origin": origin})[0], 200
        )
        self.app.status.assert_called_once_with(True)

    def test_static_paths_cannot_escape_build_and_responses_have_security_headers(self):
        for route in ("/../private.txt", "/%2e%2e/private.txt", "/%2e%2e%2fprivate.txt"):
            with self.subTest(route=route):
                status, _, body = self.request(route, authenticated=False)
                self.assertEqual(status, 400)
                self.assertNotIn(b"private fixture", body)
        status, headers, body = self.request("/", authenticated=False)
        self.assertEqual((status, body), (200, b"synthetic interface"))
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_post_rejects_malformed_or_non_object_payloads_before_search(self):
        for body, headers in (
            ("[]", {}),
            ("invalid json", {}),
            ("{}", {"Content-Type": "text/plain"}),
            ("{}", {"Content-Length": "32769"}),
        ):
            with self.subTest(body=body, headers=headers):
                self.assertEqual(self.request("/api/search", "POST", body, headers=headers)[0], 400)
        self.app.start_search.assert_not_called()
        request = {"query": "synthetic field", "top_k": 3}
        status, _, body = self.request("/api/search", "POST", json.dumps(request))
        self.assertEqual((status, json.loads(body)), (202, {"task_id": "synthetic-task"}))
        self.app.start_search.assert_called_once_with(request)

    def test_unknown_api_route_never_falls_through_to_static_files(self):
        api_directory = self.app.build / "api"
        api_directory.mkdir()
        (api_directory / "private.txt").write_text("hidden synthetic data", encoding="utf-8")
        status, _, body = self.request("/api/private.txt")
        self.assertEqual(status, 404)
        self.assertNotIn(b"hidden synthetic data", body)

    def test_shutdown_requires_auth_and_only_stops_the_test_owned_server(self):
        self.assertEqual(self.request("/api/shutdown", "POST", "{}", authenticated=False)[0], 403)
        self.assertTrue(self.thread.is_alive())
        status, _, body = self.request("/api/shutdown", "POST", "{}")
        self.assertEqual((status, json.loads(body)), (200, {"state": "closing"}))
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
