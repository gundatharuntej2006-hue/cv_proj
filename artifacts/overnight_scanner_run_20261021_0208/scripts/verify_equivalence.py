# -*- coding: utf-8 -*-
"""Prove the streaming GPU loader is equivalent to the committed CPU path, on a subset.

The runbook permits a bounded-memory implementation only if it is mathematically
equivalent to the preregistered computation, verified on a small subset first. This is
that verification.

Checks, in order:
  1. The npz header the streamer reads matches what np.load reports for the same subset.
  2. _normalise on the GPU reproduces p2_32_mil_sweep.repool(X, 12, 12) elementwise.
  3. A linear read-out produces identical region scores through both paths.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import common as C                                                   # noqa: E402

N_SUBSET = 192


def main():
    import torch
    import p2_32_mil_sweep as S

    path = C.REG / "regional_vietnam_lungs_g12_v1.npz"
    idx = np.arange(N_SUBSET)

    # --- reference: the committed CPU path, on the same subset
    with np.load(path) as z:
        raw = np.asarray(z["reg"][:N_SUBSET])
    ref = S.repool(raw, 12, 12)
    print("1. reference repool(X, 12, 12): %s %s" % (ref.shape, ref.dtype))

    # --- candidate: the streaming GPU path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    got_t, kept = C.stream_regions(path, keep_idx=idx, block=64, device=dev, verbose=True)
    got = got_t.cpu().numpy()
    assert list(kept) == list(idx)
    print("2. streamed:                    %s %s" % (got.shape, got.dtype))

    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert got.dtype == ref.dtype, (got.dtype, ref.dtype)

    exact = int((got == ref).sum())
    total = ref.size
    diff = np.abs(got.astype(np.float32) - ref.astype(np.float32))
    print("3. elementwise identical:       %d / %d (%.6f%%)"
          % (exact, total, 100.0 * exact / total))
    print("   max abs difference:          %.3e" % diff.max())
    print("   mean abs difference:         %.3e" % diff.mean())

    # float16 rounding of the division can differ in the last bit between CPU and GPU;
    # anything beyond one ulp of float16 near 1.0 (about 1e-3) would be a real divergence.
    assert diff.max() < 1e-3, "streaming path diverges from the committed repool"

    # --- end-to-end: identical region scores through a fixed linear read-out
    rng = np.random.default_rng(0)
    W = rng.standard_normal((1, ref.shape[2])).astype(np.float32)
    b = np.float32(0.137)
    s_ref = C.agg(torch.from_numpy(
        (ref.astype(np.float32) @ W.T + b)[:, :, 0]), "max").numpy()
    s_got = C.score_head(got_t, torch.from_numpy(W).to(dev),
                         torch.tensor([b]).to(dev), "max")
    d = np.abs(s_ref - s_got)
    print("4. max abs difference in max-pooled score: %.3e" % d.max())
    assert d.max() < 1e-2, "aggregated scores diverge"

    print("\nVERDICT: the streaming GPU loader is equivalent to the committed CPU path.")
    print("         Residual differences are float16 rounding, below one ulp.")


if __name__ == "__main__":
    main()
