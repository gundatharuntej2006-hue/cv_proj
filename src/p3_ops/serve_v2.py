"""Launch Atul's existing v2 service with the opt-in P3 detector bridge.

python -m p3_ops.serve_v2 --p2-root PATH --input-provider module:function
The provider returns (D08-prepared uint8 full512 image, D09 crop_box).
Use a checkout containing Atul's pipeline_v2.py and scripts/serve_v2.py.
"""
import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p2-root", type=Path, required=True)
    ap.add_argument("--input-provider", help="optional module:function override for agreed P1 preparation")
    ap.add_argument("--bundle", type=Path, help="P2 bundle directory, relative to --p2-root or absolute")
    ap.add_argument("--preload", action="store_true", help="Load P2 encoder/segmenter before serving")
    ap.add_argument("--detector-disabled", action="store_true",
                    help="Explicit classifier-only mode; no lesion count or detector referral is claimed")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    root = a.p2_root.resolve()
    if not (root / "src/p2_cls/pipeline_v2.py").is_file():
        ap.error("P2 checkout has no pipeline_v2.py; use the atul branch or its merged checkout")
    # Bind the current Role 3 packages before selecting P2's source root. Older
    # team checkouts also contain p3_* packages which must not shadow these fixes.
    import p3_det  # noqa: F401
    import p3_xai  # noqa: F401
    sys.path.insert(0, str(root / "src"))
    # Import pandas before torch on Windows (same ordering as P2's launcher).
    import pandas  # noqa: F401
    if a.bundle:
        os.environ["P2_BUNDLE"] = str(a.bundle.resolve() if a.bundle.is_absolute() else root / a.bundle)
    provider = None
    if a.input_provider:
        module, name = a.input_provider.split(":", 1)
        provider = getattr(importlib.import_module(module), name)
    if not a.detector_disabled:
        from p3_det.api import detector_info
        info = detector_info()
        if info.get("mock") or not info.get("locked"):
            ap.error("P3 service requires a real locked detector")
    spec = importlib.util.spec_from_file_location("p3_bound_p2_service", root / "scripts/serve_v2.py")
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    from p3_det.integration_v2 import install, make_input_provider
    service.p3_detector_enabled = not a.detector_disabled
    if not a.detector_disabled:
        service.p3_input_provider = provider or make_input_provider(service._bundle())
    install(service)
    from p2_serve.app import create_app
    app = create_app()
    service.register_v2(app)
    if a.preload:
        from p2_cls import pipeline_v2 as p2
        bundle = service._bundle()
        p2._embedder(bundle)
        p2._segmenter(bundle)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
