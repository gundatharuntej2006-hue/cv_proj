# -*- coding: utf-8 -*-
"""Verify the remote-ZIP reader against a local server that honours Range requests.

The ZIP64 branch cannot be exercised here without building a >4 GB archive, so it is
covered only when run against the real CheXpert distribution. Everything else — locating
the central directory from the tail, parsing it, and inflating one member from its byte
range alone — is tested properly.
"""
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from p2_gate import remote_zip as RZ


class RangeHandler(BaseHTTPRequestHandler):
    """Minimal static server with Range support; the stdlib one has none."""
    payload = b""

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", extra=None):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            return self._send(200, self.payload)
        start, end = rng.split("=")[1].split("-")
        start = int(start)
        end = int(end) if end else len(self.payload) - 1
        end = min(end, len(self.payload) - 1)
        chunk = self.payload[start:end + 1]
        self._send(206, chunk, {"Content-Range": "bytes %d-%d/%d"
                                % (start, end, len(self.payload))})


class NoRangeHandler(RangeHandler):
    """A server that ignores Range entirely, like a plain file host."""
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def do_GET(self):
        self._send(200, self.payload)


FILES = {
    "imgs/patient001/view1_frontal.png": b"FRONTAL-" + b"a" * 5000,
    "imgs/patient001/view2_lateral.png": b"LATERAL-" + b"b" * 7000,
    "imgs/patient002/view1_lateral.png": b"LATERAL2-" + b"c" * 3000,
    "train.csv": b"Path,Frontal/Lateral\nimgs/patient001/view2_lateral.png,Lateral\n",
}


def build_zip(tmp_path):
    p = tmp_path / "archive.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in FILES.items():
            z.writestr(name, data)
    return p.read_bytes()


@pytest.fixture
def server(tmp_path):
    RangeHandler.payload = build_zip(tmp_path)
    srv = HTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d/archive.zip" % srv.server_port
    srv.shutdown()


@pytest.fixture
def server_no_range(tmp_path):
    NoRangeHandler.payload = build_zip(tmp_path)
    srv = HTTPServer(("127.0.0.1", 0), NoRangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d/archive.zip" % srv.server_port
    srv.shutdown()


def test_probe_reports_range_support(server):
    info = RZ.probe(server)
    assert info["range_supported"] is True
    assert info["size"] == len(RangeHandler.payload)


def test_probe_detects_a_server_without_range(server_no_range):
    assert RZ.probe(server_no_range)["range_supported"] is False


def test_central_directory_lists_every_member(server):
    members = RZ.central_directory(server)
    assert set(members) == set(FILES)
    m = members["imgs/patient001/view2_lateral.png"]
    assert m.file_size == len(FILES["imgs/patient001/view2_lateral.png"])
    assert m.compressed_size < m.file_size        # it really was deflated


def test_fetching_one_member_returns_exactly_that_file(server):
    members = RZ.central_directory(server)
    for name, expected in FILES.items():
        assert RZ.fetch(server, members[name]) == expected


def test_fetching_one_member_transfers_far_less_than_the_archive(server, monkeypatch):
    """The whole point: pulling one file must not move the whole archive."""
    members = RZ.central_directory(server)
    moved = {"bytes": 0}
    real = RZ._get

    def counting(url, start, end, headers=None):
        b = real(url, start, end, headers)
        moved["bytes"] += len(b)
        return b

    monkeypatch.setattr(RZ, "_get", counting)
    name = "imgs/patient001/view2_lateral.png"
    data = RZ.fetch(server, members[name])
    assert data == FILES[name]
    assert moved["bytes"] < members[name].compressed_size + 200
    assert moved["bytes"] < len(RangeHandler.payload) / 2


def test_a_server_without_range_fails_loudly(server_no_range):
    with pytest.raises(RZ.RangeError, match="Range"):
        RZ.central_directory(server_no_range)


def test_fetch_many_writes_the_requested_files(server, tmp_path):
    members = RZ.central_directory(server)
    wanted = [members[n] for n in FILES if n.endswith("_lateral.png")]
    written = RZ.fetch_many(server, wanted, tmp_path / "out", log=lambda *_: None)
    assert len(written) == 2
    for p in written:
        assert p.read_bytes().startswith(b"LATERAL")


def test_fetch_many_resumes_and_does_not_refetch(server, tmp_path, monkeypatch):
    """A run killed by a dropped connection must continue, not start over."""
    members = RZ.central_directory(server)
    wanted = [members[n] for n in sorted(FILES)]
    out = tmp_path / "out"
    RZ.fetch_many(server, wanted[:2], out, log=lambda *_: None)
    assert len(list(out.glob("*"))) == 2

    calls = {"n": 0}
    real = RZ.fetch

    def counting(url, m, headers=None):
        calls["n"] += 1
        return real(url, m, headers)

    monkeypatch.setattr(RZ, "fetch", counting)
    written = RZ.fetch_many(server, wanted, out, log=lambda *_: None)
    assert len(written) == len(wanted)
    assert calls["n"] == len(wanted) - 2, "already-complete files were fetched again"


def test_fetch_many_survives_one_bad_member(server, tmp_path, monkeypatch):
    """One failure must not abandon the other 499 files."""
    members = RZ.central_directory(server)
    wanted = [members[n] for n in sorted(FILES)]
    real = RZ.fetch

    def flaky(url, m, headers=None):
        if m.name == sorted(FILES)[1]:
            raise OSError("simulated connection reset")
        return real(url, m, headers)

    monkeypatch.setattr(RZ, "fetch", flaky)
    written = RZ.fetch_many(server, wanted, tmp_path / "out", log=lambda *_: None)
    assert len(written) == len(wanted) - 1


def test_a_partial_write_is_never_mistaken_for_a_finished_file(server, tmp_path):
    members = RZ.central_directory(server)
    name = sorted(FILES)[0]
    out = tmp_path / "out"
    out.mkdir()
    truncated = out / name.replace("/", "__")
    truncated.write_bytes(b"half")                 # wrong size on disk
    RZ.fetch_many(server, [members[name]], out, log=lambda *_: None)
    assert truncated.read_bytes() == FILES[name], "a short file must be re-fetched"
