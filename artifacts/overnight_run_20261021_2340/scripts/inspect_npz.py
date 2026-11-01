"""Read npz member shapes/dtypes from the zip headers without loading arrays."""
import sys, zipfile, numpy as np


def describe(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".npy"):
                out[name] = {"note": "non-array member"}
                continue
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            out[name[:-4]] = {"shape": list(shape), "dtype": str(dtype)}
    return out


if __name__ == "__main__":
    import json
    for p in sys.argv[1:]:
        print(p)
        print(json.dumps(describe(p), indent=2))
