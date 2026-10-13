# -*- coding: utf-8 -*-
"""Read selected members out of a remote ZIP without downloading the whole archive.

Why this exists: CheXpert is distributed as a single 471 GB zip, and P2-10 needs about
200 lateral images out of its 224,316. The project drive is 392 GB in total, so the
archive cannot be stored even if the drive were emptied.

A ZIP is randomly accessible by design. Its index — the central directory — sits at the
*end* of the file, and each member is an independently compressed blob at a known offset.
So if the server honours HTTP Range requests, the whole archive need never be transferred:

    1. read the last ~128 KB to find the End Of Central Directory record,
    2. follow it (via the ZIP64 locator, which a 471 GB archive must use) to the central
       directory, and read that,
    3. for each wanted member, read only its byte range and inflate it.

This is ordinary HTTP, not a workaround: Range is a standard feature and the data is used
under the licence already accepted. It simply avoids moving 471 GB to obtain 200 files.

If the server does not support Range, `probe` says so and the idea is dead — fall back to
another source rather than trying to stream the archive sequentially, which would transfer
the whole thing anyway.
"""
import struct
import time
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CEN_SIG = b"PK\x01\x02"
LOC_SIG = b"PK\x03\x04"


@dataclass
class Member:
    name: str
    compress_type: int
    compressed_size: int
    file_size: int
    header_offset: int


class RangeError(RuntimeError):
    pass


def _get(url: str, start: int, end: int, headers: Optional[dict] = None,
         retries: int = 5) -> bytes:
    """Inclusive byte range, as HTTP defines it.

    Retried with backoff: several hundred sequential HTTPS requests over a domestic
    connection will hit a timeout sooner or later, and losing an hour of fetching to one
    dropped packet is not acceptable.
    """
    h = dict(headers or {})
    h["Range"] = "bytes=%d-%d" % (start, end)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RangeError("server ignored the Range header (HTTP %s); it would "
                                     "send the whole archive" % r.status)
                return r.read()
        except RangeError:
            raise                                   # not transient: do not retry
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError("range request failed after %d attempts: %s" % (retries, last))


def probe(url: str, headers: Optional[dict] = None) -> dict:
    """Does this URL support random access, and how big is it?"""
    req = urllib.request.Request(url, headers=dict(headers or {}), method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers.get("Content-Length") or 0)
        accepts = (r.headers.get("Accept-Ranges") or "").lower()
    ok = accepts == "bytes"
    if not ok:                      # some servers omit the header but honour the request
        try:
            ok = len(_get(url, 0, 1, headers)) == 2
        except Exception:
            ok = False
    return {"size": size, "range_supported": bool(ok)}


def central_directory(url: str, headers: Optional[dict] = None) -> Dict[str, Member]:
    """Member index, read from the tail of the archive."""
    info = probe(url, headers)
    if not info["range_supported"]:
        raise RangeError("this server does not support Range requests")
    size = info["size"]
    if not size:
        raise RangeError("no Content-Length; cannot locate the central directory")

    tail = _get(url, max(size - 131072, 0), size - 1, headers)
    i = tail.rfind(EOCD_SIG)
    if i < 0:
        raise RangeError("no End Of Central Directory record in the last 128 KB")

    cd_size = struct.unpack("<I", tail[i + 12:i + 16])[0]
    cd_off = struct.unpack("<I", tail[i + 16:i + 20])[0]

    j = tail.rfind(EOCD64_LOCATOR_SIG)
    if j >= 0:                       # ZIP64: anything past 4 GB must use it
        eocd64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        head = _get(url, eocd64_off, eocd64_off + 55, headers)
        if head[:4] != EOCD64_SIG:
            raise RangeError("ZIP64 locator does not point at an EOCD64 record")
        cd_size = struct.unpack("<Q", head[40:48])[0]
        cd_off = struct.unpack("<Q", head[48:56])[0]

    cd = _get(url, cd_off, cd_off + cd_size - 1, headers)
    members, p = {}, 0
    while p + 46 <= len(cd) and cd[p:p + 4] == CEN_SIG:
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize = struct.unpack("<I", cd[p + 20:p + 24])[0]
        usize = struct.unpack("<I", cd[p + 24:p + 28])[0]
        n_len, x_len, c_len = struct.unpack("<HHH", cd[p + 28:p + 34])
        lho = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + n_len].decode("utf-8", "replace")
        extra = cd[p + 46 + n_len:p + 46 + n_len + x_len]

        if 0xFFFFFFFF in (csize, usize, lho):      # ZIP64 extra field
            q = 0
            while q + 4 <= len(extra):
                tag, ln = struct.unpack("<HH", extra[q:q + 4])
                if tag == 0x0001:
                    vals, r = [], q + 4
                    for target in (usize, csize, lho):
                        if target == 0xFFFFFFFF and r + 8 <= q + 4 + ln:
                            vals.append(struct.unpack("<Q", extra[r:r + 8])[0]); r += 8
                        else:
                            vals.append(None)
                    if vals[0] is not None: usize = vals[0]
                    if vals[1] is not None: csize = vals[1]
                    if vals[2] is not None: lho = vals[2]
                    break
                q += 4 + ln
        members[name] = Member(name, method, csize, usize, lho)
        p += 46 + n_len + x_len + c_len
    return members


def fetch(url: str, member: Member, headers: Optional[dict] = None) -> bytes:
    """One member's bytes, decompressed. Reads only that member's range."""
    head = _get(url, member.header_offset, member.header_offset + 29, headers)
    if head[:4] != LOC_SIG:
        raise RangeError("no local file header at offset %d" % member.header_offset)
    n_len, x_len = struct.unpack("<HH", head[26:30])
    start = member.header_offset + 30 + n_len + x_len
    raw = _get(url, start, start + member.compressed_size - 1, headers)
    if member.compress_type == 0:
        return raw
    if member.compress_type == 8:
        return zlib.decompress(raw, -15)
    raise RangeError("unsupported compression method %d for %s"
                     % (member.compress_type, member.name))


def fetch_many(url: str, members: List[Member], out_dir, headers=None, log=print,
               resume: bool = True):
    """Save several members, flattened into out_dir. Returns the written paths.

    Resumable: a member already on disk at its expected size is skipped, so a run killed
    by a dropped connection continues where it stopped rather than starting over. Each
    file is written to a temporary name and renamed, so a partial write is never mistaken
    for a finished one.
    """
    import os
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written, got, skipped, failed = [], 0, 0, []
    for i, m in enumerate(members, 1):
        dest = out / m.name.replace("/", "__").replace("\\", "__")
        if resume and dest.exists() and dest.stat().st_size == m.file_size:
            written.append(dest); skipped += 1
            continue
        try:
            data = fetch(url, m, headers)
        except Exception as e:                      # one bad member must not end the run
            failed.append((m.name, str(e)[:80]))
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        written.append(dest)
        got += len(data)
        if i % 25 == 0 or i == len(members):
            log("  %d/%d  fetched %.0f MB, skipped %d%s"
                % (i, len(members), got / 1e6, skipped,
                   ", %d failed" % len(failed) if failed else ""))
    if failed:
        log("  %d members could not be fetched; re-run to retry them:" % len(failed))
        for name, err in failed[:5]:
            log("    %s -> %s" % (name, err))
    return written
