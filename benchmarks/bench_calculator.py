"""Benchmark for the end-to-end calculate_anomaly() interface."""

import timeit

import numpy as np

from forward_model.compute.calculator import calculate_anomaly
from forward_model.models.body import GeologicBody
from forward_model.models.field import MagneticField
from forward_model.models.model import ForwardModel


def _make_rect_body(
    x_center: float,
    z_top: float,
    z_bot: float,
    half_width: float,
    name: str,
    susceptibility: float = 0.05,
) -> GeologicBody:
    return GeologicBody(
        vertices=[
            [x_center - half_width, z_top],
            [x_center + half_width, z_top],
            [x_center + half_width, z_bot],
            [x_center - half_width, z_bot],
        ],
        susceptibility=susceptibility,
        name=name,
    )


_FIELD = MagneticField(intensity=50000.0, inclination=60.0, declination=0.0)


def _make_model(n_bodies: int, n_obs: int) -> ForwardModel:
    bodies = [
        _make_rect_body(
            x_center=i * 200.0,
            z_top=100.0 + i * 20.0,
            z_bot=300.0 + i * 20.0,
            half_width=50.0,
            name=f"body_{i}",
        )
        for i in range(n_bodies)
    ]
    obs_x = list(np.linspace(-1000.0, 1000.0 + n_bodies * 200.0, n_obs))
    return ForwardModel(
        bodies=bodies,
        field=_FIELD,
        observation_x=obs_x,
        observation_z=0.0,
    )


_CASES: list[tuple[str, int, int]] = [
    ("simple   (1 body,   201 obs)", 1, 201),
    ("multi    (6 bodies, 201 obs)", 6, 201),
    ("large    (20 bodies, 1000 obs)", 20, 1000),
]

_REPEATS = 5
_NUMBER = 10


def main() -> None:
    print("=" * 60)
    print("bench_calculator: calculate_anomaly (serial vs. parallel)")
    print("=" * 60)
    for label, n_bodies, n_obs in _CASES:
        model = _make_model(n_bodies, n_obs)

        # Warm-up
        calculate_anomaly(model)

        serial_times = timeit.repeat(
            lambda m=model: calculate_anomaly(m, parallel=False),
            number=_NUMBER,
            repeat=_REPEATS,
        )
        serial_ms = min(serial_times) / _NUMBER * 1000

        parallel_times = timeit.repeat(
            lambda m=model: calculate_anomaly(m, parallel=True),
            number=_NUMBER,
            repeat=_REPEATS,
        )
        parallel_ms = min(parallel_times) / _NUMBER * 1000

        speedup = serial_ms / parallel_ms if parallel_ms > 0 else float("inf")
        print(f"  {label}")
        print(f"    serial:   {serial_ms:.3f} ms")
        print(f"    parallel: {parallel_ms:.3f} ms  (speedup {speedup:.2f}×)")
    print()


if __name__ == "__main__":
    main()
