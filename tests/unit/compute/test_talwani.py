"""Tests for Talwani algorithm functions."""

import numpy as np

from forward_model.compute import compute_polygon_anomaly, field_to_magnetization


class TestFieldToMagnetization:
    """Tests for field to magnetization conversion."""

    def test_vertical_field(self) -> None:
        """Test with vertical inducing field (inclination = 90)."""
        mag = field_to_magnetization(
            susceptibility=0.1,
            field_intensity=50000.0,  # nT
            field_inclination=90.0,
            field_declination=0.0,
        )
        assert mag.shape == (2,)
        # Vertical field: Mx should be ~0, Mz should be negative (down)
        assert np.abs(mag[0]) < 1e-10  # Mx ~ 0
        assert mag[1] < 0  # Mz negative (field points down)

    def test_horizontal_field(self) -> None:
        """Test with horizontal inducing field (inclination = 0)."""
        mag = field_to_magnetization(
            susceptibility=0.1,
            field_intensity=50000.0,
            field_inclination=0.0,
            field_declination=0.0,
        )
        assert mag.shape == (2,)
        # Horizontal field: Mx non-zero, Mz should be ~0
        assert mag[0] > 0  # Mx positive
        assert np.abs(mag[1]) < 1e-10  # Mz ~ 0

    def test_zero_susceptibility(self) -> None:
        """Test that zero susceptibility gives zero magnetization."""
        mag = field_to_magnetization(
            susceptibility=0.0,
            field_intensity=50000.0,
            field_inclination=60.0,
            field_declination=0.0,
        )
        assert np.allclose(mag, [0.0, 0.0])

    def test_zero_remanence_unchanged(self) -> None:
        """With all remanent params at default, result equals induced-only result."""
        induced_only = field_to_magnetization(
            susceptibility=0.05,
            field_intensity=50000.0,
            field_inclination=60.0,
            field_declination=0.0,
        )
        with_zero_remanence = field_to_magnetization(
            susceptibility=0.05,
            field_intensity=50000.0,
            field_inclination=60.0,
            field_declination=0.0,
            remanent_intensity=0.0,
            remanent_inclination=0.0,
            remanent_declination=0.0,
        )
        assert np.allclose(induced_only, with_zero_remanence)

    def test_pure_remanent_body(self) -> None:
        """With susceptibility=0, result equals the projected remanent vector."""
        rem_intensity = 3.0  # A/m
        rem_inc = 45.0  # degrees
        rem_dec = 0.0
        mag = field_to_magnetization(
            susceptibility=0.0,
            field_intensity=50000.0,
            field_inclination=60.0,
            field_declination=0.0,
            remanent_intensity=rem_intensity,
            remanent_inclination=rem_inc,
            remanent_declination=rem_dec,
        )
        expected_mx = (
            rem_intensity * np.cos(np.deg2rad(rem_inc)) * np.cos(np.deg2rad(rem_dec))
        )
        expected_mz = -rem_intensity * np.sin(np.deg2rad(rem_inc))
        assert np.allclose(mag, [expected_mx, expected_mz])

    def test_vertical_remanence_adds_correctly(self) -> None:
        """remanent_inclination=90 gives Mx_rem≈0, Mz_rem<0."""
        mag = field_to_magnetization(
            susceptibility=0.0,
            field_intensity=50000.0,
            field_inclination=60.0,
            field_declination=0.0,
            remanent_intensity=2.0,
            remanent_inclination=90.0,
            remanent_declination=0.0,
        )
        assert np.abs(mag[0]) < 1e-10  # Mx ~ 0
        assert mag[1] < 0  # Mz negative (pointing downward)
        assert np.isclose(mag[1], -2.0)

    def test_antiparallel_remanence_reduces_anomaly(self) -> None:
        """Remanent vector antiparallel to induced gives total |M| < induced |M|."""
        inc = 60.0
        dec = 0.0
        susceptibility = 0.05
        intensity = 50000.0

        induced = field_to_magnetization(
            susceptibility=susceptibility,
            field_intensity=intensity,
            field_inclination=inc,
            field_declination=dec,
        )

        # Antiparallel remanence: opposite inclination, same declination
        mag_total = field_to_magnetization(
            susceptibility=susceptibility,
            field_intensity=intensity,
            field_inclination=inc,
            field_declination=dec,
            remanent_intensity=0.5,
            remanent_inclination=-inc,
            remanent_declination=dec,
        )

        assert np.linalg.norm(mag_total) < np.linalg.norm(induced)


class TestComputePolygonAnomaly:
    """Tests for Talwani polygon anomaly computation."""

    def test_simple_rectangle(self) -> None:
        """Test anomaly from a simple rectangular body."""
        # Rectangle centered at x=25, depth 100-200m
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )

        # Observation points along surface
        obs = np.array(
            [[-100.0, 0.0], [0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [100.0, 0.0]],
            dtype=np.float64,
        )

        # Vertical magnetization (simple case)
        mag = np.array([0.0, -1000.0], dtype=np.float64)  # A/m

        anomaly = compute_polygon_anomaly(vertices, obs, mag)

        # Basic checks
        assert anomaly.shape == (5,)
        assert np.all(np.isfinite(anomaly))

        # Anomaly should be largest near center of body
        center_idx = 2  # x=25, directly above center
        # Greater than far field
        assert np.abs(anomaly[center_idx]) > np.abs(anomaly[0])

    def test_symmetry(self) -> None:
        """Test that symmetric body produces symmetric anomaly."""
        # Rectangle centered at x=0
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )

        # Symmetric observation points
        obs = np.array(
            [[-100.0, 0.0], [-50.0, 0.0], [0.0, 0.0], [50.0, 0.0], [100.0, 0.0]],
            dtype=np.float64,
        )

        # Vertical magnetization
        mag = np.array([0.0, -1000.0], dtype=np.float64)

        anomaly = compute_polygon_anomaly(vertices, obs, mag)

        # Check symmetry: anomaly[-100] ≈ anomaly[100], etc.
        assert np.allclose(anomaly[0], anomaly[4], rtol=1e-10)  # x=-100 vs x=100
        assert np.allclose(anomaly[1], anomaly[3], rtol=1e-10)  # x=-50 vs x=50

    def test_zero_magnetization(self) -> None:
        """Test that zero magnetization gives zero anomaly."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        obs = np.array([[0.0, 0.0], [50.0, 0.0]], dtype=np.float64)
        mag = np.array([0.0, 0.0], dtype=np.float64)

        anomaly = compute_polygon_anomaly(vertices, obs, mag)
        assert np.allclose(anomaly, 0.0)

    def test_far_field_decay(self) -> None:
        """Test that anomaly decreases with distance."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )

        # Observation points at increasing distances
        obs = np.array(
            [[25.0, 0.0], [125.0, 0.0], [500.0, 0.0], [1000.0, 0.0]],
            dtype=np.float64,
        )
        mag = np.array([0.0, -1000.0], dtype=np.float64)

        anomaly = compute_polygon_anomaly(vertices, obs, mag)

        # Anomaly should decay with distance
        assert np.abs(anomaly[0]) > np.abs(anomaly[1])
        assert np.abs(anomaly[1]) > np.abs(anomaly[2])
        assert np.abs(anomaly[2]) > np.abs(anomaly[3])

    def test_degenerate_edge(self) -> None:
        """Test handling of degenerate edges (same vertex repeated)."""
        # Include a duplicate vertex
        vertices = np.array(
            [
                [0.0, 100.0],
                [50.0, 100.0],
                [50.0, 100.0],
                [50.0, 200.0],
                [0.0, 200.0],
            ],
            dtype=np.float64,
        )
        obs = np.array([[25.0, 0.0]], dtype=np.float64)
        mag = np.array([0.0, -1000.0], dtype=np.float64)

        # Should not raise an error
        anomaly = compute_polygon_anomaly(vertices, obs, mag)
        assert np.isfinite(anomaly[0])


class TestVectorizedVsScalar:
    """Regression tests: vectorized implementation vs. scalar reference."""

    @staticmethod
    def _scalar_compute(
        vertices: np.ndarray,
        observation_points: np.ndarray,
        magnetization: np.ndarray,
        min_distance: float = 1e-10,
    ) -> np.ndarray:
        """Scalar reference implementation (original nested-loop version)."""
        n_vertices = len(vertices)
        n_obs = len(observation_points)
        anomaly = np.zeros(n_obs, dtype=np.float64)

        mu_0_4pi_nT = 1e-7 * 1e9
        Mx, Mz = magnetization

        for i, obs in enumerate(observation_points):
            contribution = 0.0
            for j in range(n_vertices):
                j_next = (j + 1) % n_vertices
                x1, z1 = vertices[j] - obs
                x2, z2 = vertices[j_next] - obs

                if np.abs(x2 - x1) < min_distance and np.abs(z2 - z1) < min_distance:
                    continue

                r1 = np.sqrt(x1**2 + z1**2)
                r2 = np.sqrt(x2**2 + z2**2)

                if r1 < min_distance or r2 < min_distance:
                    continue

                theta1 = np.arctan2(z1, x1)
                theta2 = np.arctan2(z2, x2)
                dtheta = theta2 - theta1
                if dtheta > np.pi:
                    dtheta -= 2 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2 * np.pi

                dx = x2 - x1
                dz = z2 - z1
                edge_length = np.sqrt(dx**2 + dz**2)
                if edge_length < min_distance:
                    continue

                tx = dx / edge_length
                tz = dz / edge_length

                log_term = np.log(r2 / r1)
                contribution += Mx * (dtheta * tz - log_term * tx) + Mz * (
                    -dtheta * tx - log_term * tz
                )

            anomaly[i] = mu_0_4pi_nT * contribution

        return anomaly

    def _assert_close(
        self,
        vertices: np.ndarray,
        obs: np.ndarray,
        mag: np.ndarray,
    ) -> None:
        scalar = self._scalar_compute(vertices, obs, mag)
        vectorized = compute_polygon_anomaly(vertices, obs, mag)
        max_diff = float(np.max(np.abs(vectorized - scalar)))
        assert max_diff < 1e-10, f"Max abs diff {max_diff} >= 1e-10"

    def test_simple_rectangle(self) -> None:
        """Vectorized matches scalar on a simple rectangle."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        obs = np.array(
            [[-100.0, 0.0], [0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [100.0, 0.0]],
            dtype=np.float64,
        )
        mag = np.array([0.0, -1000.0], dtype=np.float64)
        self._assert_close(vertices, obs, mag)

    def test_symmetric_body(self) -> None:
        """Vectorized matches scalar on a symmetric body."""
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs = np.array(
            [[-100.0, 0.0], [-50.0, 0.0], [0.0, 0.0], [50.0, 0.0], [100.0, 0.0]],
            dtype=np.float64,
        )
        mag = np.array([500.0, -800.0], dtype=np.float64)
        self._assert_close(vertices, obs, mag)

    def test_dense_grid(self) -> None:
        """Vectorized matches scalar on a dense 201-point observation grid."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        xs = np.linspace(-500.0, 500.0, 201)
        obs = np.column_stack([xs, np.zeros(201)])
        mag = np.array([300.0, -1000.0], dtype=np.float64)
        self._assert_close(vertices, obs, mag)

    def test_degenerate_edge_matches(self) -> None:
        """Vectorized matches scalar even with a degenerate (duplicate) edge."""
        vertices = np.array(
            [
                [0.0, 100.0],
                [50.0, 100.0],
                [50.0, 100.0],
                [50.0, 200.0],
                [0.0, 200.0],
            ],
            dtype=np.float64,
        )
        obs = np.array([[25.0, 0.0], [-50.0, 0.0], [200.0, 0.0]], dtype=np.float64)
        mag = np.array([0.0, -1000.0], dtype=np.float64)
        self._assert_close(vertices, obs, mag)
