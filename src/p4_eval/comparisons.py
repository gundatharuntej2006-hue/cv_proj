"""Section 14 comparison table generation.

The legacy table contained illustrative constants and has no traceable run inputs.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


def generate_table_14_comparisons(out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Refuse to publish comparison results until a provenance-backed runner exists."""
    raise RuntimeError(
        "Table 14 has no verified comparison artifacts; a provenance-backed "
        "generator is required before publishing results"
    )
