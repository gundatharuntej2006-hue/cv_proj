# -*- coding: utf-8 -*-
"""Run the inference API for the frontend to call.

    py -3 scripts/serve.py                  # http://127.0.0.1:8000  (docs at /docs)
    py -3 scripts/serve.py --port 8080
    py -3 scripts/serve.py --preload         # load the model now rather than on first request

The first request pays for the backbone load (a few seconds), so `--preload` is worth it
before a demo.

It binds to localhost and has **no authentication**. `--host` can change that but should
not: this serves medical images and stores nothing, which is only defensible because it is
not reachable from a network.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--reload", action="store_true", help="auto-restart on code changes")
    args = ap.parse_args()

    import uvicorn
    from p2_serve import app as A

    if args.host != "127.0.0.1":
        print("WARNING: binding to %s exposes an unauthenticated medical-image endpoint"
              % args.host)
    if args.preload:
        info = A.model_info()
        print("loaded %s (%s)%s" % (info["model_id"], info["variant"],
                                    "  [PROVISIONAL]" if info["provisional"] else ""))

    print("http://%s:%d/docs" % (args.host, args.port))
    uvicorn.run(A.create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
