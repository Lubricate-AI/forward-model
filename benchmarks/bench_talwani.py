"""Benchmark for the core Talwani algorithm (compute_polygon_anomaly)."""

import timeit

import numpy as np

from forward_model.compute.talwani import compute_polygon_anomaly

# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def _make_inputs(
    n_obs: int, n_vertices: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-1000.0, 1000.0, n_obs)
    obs = np.column_stack([xs, np.zeros(n_obs)])
    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    verts = np.column_stack([200 * np.cos(angles), 300 + 100 * np.sin(angles)])
    mag = np.array([300.0, -1000.0], dtype=np.float64)
    return verts, obs, mag


_CASES: list[tuple[str, int, int]] = [
    ("baseline (201 obs, 4 vertices)", 201, 4),
    ("large    (1000 obs, 8 vertices)", 1000, 8),
]

_REPEATS = 5
_NUMBER = 20


def main() -> None:
    print("=" * 60)
    print("bench_talwani: compute_polygon_anomaly (vectorized)")
    print("=" * 60)
    for label, n_obs, n_verts in _CASES:
        verts, obs, mag = _make_inputs(n_obs, n_verts)

        # Warm-up
        compute_polygon_anomaly(verts, obs, mag)

        elapsed = timeit.repeat(
            lambda v=verts, o=obs, m=mag: compute_polygon_anomaly(v, o, m),
            number=_NUMBER,
            repeat=_REPEATS,
        )
        best_ms = min(elapsed) / _NUMBER * 1000
        print(f"  {label}: {best_ms:.3f} ms / call (best of {_REPEATS}×{_NUMBER})")
    print()


if __name__ == "__main__":
    main()
