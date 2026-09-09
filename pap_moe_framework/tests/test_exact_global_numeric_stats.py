from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "recompute_lerobot_numeric_stats.py"
)
SPEC = importlib.util.spec_from_file_location("exact_global_stats", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_recomputes_union_quantiles_instead_of_averaging_episode_quantiles(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    # Two disjoint episode modes. Averaging their q99 values would be near 5,
    # while the q99 of their union is near 10.
    values = np.concatenate(
        [np.linspace(0.0, 1.0, 100), np.linspace(10.0, 11.0, 100)]
    ).astype(np.float32)
    actions = [[float(value), -1.5707] for value in values]
    pq.write_table(
        pa.table({"action": actions}),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "action": {"dtype": "float32", "shape": [2]},
                }
            }
        )
    )
    (root / "meta" / "stats.json").write_text(
        json.dumps({"action": {"q01": [0.5, -1.5707], "q99": [10.5, -1.5707]}})
    )

    MODULE.recompute_numeric_stats(root)

    stats = json.loads((root / "meta" / "stats.json").read_text())["action"]
    np.testing.assert_allclose(stats["q01"], np.quantile(np.asarray(actions), 0.01, axis=0))
    np.testing.assert_allclose(stats["q99"], np.quantile(np.asarray(actions), 0.99, axis=0))
    np.testing.assert_allclose(stats["std"], np.std(np.asarray(actions, dtype=np.float64), axis=0))
    assert stats["q99"][0] > 10.9
