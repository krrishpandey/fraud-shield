import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fraudshield import desktop


def test_find_free_port_returns_bindable_port():
    port = desktop.find_free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


@pytest.mark.parametrize(
    "has_cuda,has_laya,expected",
    [(True, True, "auto"), (False, True, "cached"), (True, False, "cached"), (False, False, "cached")],
)
def test_choose_laya_mode_auto(has_cuda, has_laya, expected):
    assert desktop.choose_laya_mode("auto", has_cuda=has_cuda, has_laya=has_laya) == expected


def test_choose_laya_mode_explicit_overrides_auto():
    assert desktop.choose_laya_mode("cpu", has_cuda=False, has_laya=True) == "cpu"


def test_choose_laya_mode_local_without_laya_falls_back_to_cached():
    assert desktop.choose_laya_mode("local", has_cuda=True, has_laya=False) == "cached"


def test_wait_for_server_true_when_health_responds():
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert desktop.wait_for_server(f"http://127.0.0.1:{srv.server_port}/health", timeout=5)
    finally:
        srv.shutdown()


def test_wait_for_server_false_on_timeout():
    port = desktop.find_free_port()
    assert desktop.wait_for_server(f"http://127.0.0.1:{port}/health", timeout=0.5) is False


def test_load_config_applies_laya_mode_and_keeps_components(tmp_path):
    cfg = desktop.load_config(mode="cached")
    assert cfg["laya"]["mode"] == "cached"
    assert cfg["components"]["featurizer"]  # real components stay wired


def test_load_config_honours_fs_config(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("laya:\n  mode: local\ncomponents: {featurizer: x:y}\n")
    monkeypatch.setenv("FS_CONFIG", str(p))
    cfg = desktop.load_config(mode="cached")
    assert cfg["laya"]["mode"] == "cached" and cfg["components"]["featurizer"] == "x:y"


def test_window_url_uses_localhost_for_passkeys():
    assert desktop.window_url(8123) == "http://localhost:8123"


def test_localhost_reaches_a_server_bound_to_127_0_0_1_only():
    """The window opens http://localhost:<port> while uvicorn binds 127.0.0.1: localhost must still connect even when
    it resolves to ::1 first (refused there, then 127.0.0.1)."""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert desktop.wait_for_server(f"{desktop.window_url(srv.server_port)}/health", timeout=5)
    finally:
        srv.shutdown()


def test_loopback_sockets_serve_localhost_on_both_stacks_and_never_the_network():
    import time
    import urllib.request

    import uvicorn

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    port = desktop.find_free_port()
    socks = desktop.loopback_sockets(port)
    assert {s.getsockname()[0] for s in socks} <= {"127.0.0.1", "::1"}
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    threading.Thread(target=server.run, kwargs={"sockets": socks}, daemon=True).start()
    try:
        assert desktop.wait_for_server(f"http://127.0.0.1:{port}/", timeout=10)
        t = time.perf_counter()
        with urllib.request.urlopen(desktop.window_url(port) + "/", timeout=5) as r:
            assert r.status == 200
        if len(socks) == 2:  # IPv6 loopback present: no refused-::1 detour
            assert time.perf_counter() - t < 0.5
    finally:
        server.should_exit = True
