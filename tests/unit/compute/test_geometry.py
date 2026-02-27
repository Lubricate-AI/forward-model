"""Unit tests for the private Talwani edge-geometry kernels."""

import numpy as np

from forward_model.compute.geometry import (
    edge_geometry_2_5d,
    edge_geometry_2_75d,
    edge_geometry_2d,
)


class TestEdgeGeometry2D:
    """Tests for edge_geometry_2d."""

    # Unit square (CCW winding): vertices go counter-clockwise
    _SQUARE = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float64
    )

    def test_winding_angle_interior_point(self) -> None:
        """sum(dtheta) ≈ ±2π for an observation point inside the polygon."""
        obs = np.array([[1.0, 1.0]], dtype=np.float64)  # interior centroid

        total_dtheta = sum(
            float(dtheta[0])
            for _, _, dtheta, _, _ in edge_geometry_2d(self._SQUARE, obs)
        )
        assert np.isclose(abs(total_dtheta), 2 * np.pi, atol=1e-10), (
            f"Interior winding angle {total_dtheta:.6f} is not ±2π"
        )

    def test_winding_angle_exterior_point(self) -> None:
        """sum(dtheta) ≈ 0 for an observation point outside the polygon."""
        obs = np.array([[10.0, 10.0]], dtype=np.float64)  # far exterior

        total_dtheta = sum(
            float(dtheta[0])
            for _, _, dtheta, _, _ in edge_geometry_2d(self._SQUARE, obs)
        )
        assert np.isclose(total_dtheta, 0.0, atol=1e-10), (
            f"Exterior winding angle {total_dtheta:.6f} is not 0"
        )

    def test_degenerate_edge_skipped(self) -> None:
        """A zero-length (duplicate) edge is skipped; one fewer edge is yielded."""
        # 5 vertices with one degenerate edge (vertex 1 == vertex 2)
        vertices = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 0.0],  # duplicate → degenerate edge
                [2.0, 2.0],
                [0.0, 2.0],
            ],
            dtype=np.float64,
        )
        obs = np.array([[1.0, 1.0]], dtype=np.float64)

        edges = list(edge_geometry_2d(vertices, obs))
        assert len(edges) == 4, f"Expected 4 non-degenerate edges, got {len(edges)}"

    def test_log_ratio_correct(self) -> None:
        """log_ratio for a known edge matches hand-computed log(r2/r1)."""
        # Triangle; first edge: vertex (0,1) → vertex (1,2)
        # obs at origin: x1=0,z1=1 → r1=1; x2=1,z2=2 → r2=sqrt(5)
        vertices = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
        obs = np.array([[0.0, 0.0]], dtype=np.float64)

        first_edge = next(iter(edge_geometry_2d(vertices, obs)))
        _, _, _, log_ratio, valid = first_edge

        assert valid[0], "Observation point should be valid for this edge"
        expected = np.log(np.sqrt(5.0) / 1.0)
        assert np.isclose(log_ratio[0], expected, atol=1e-12), (
            f"log_ratio {log_ratio[0]:.12f} != expected {expected:.12f}"
        )

    def test_yields_four_edges_for_square(self) -> None:
        """A square polygon yields exactly 4 edges (no degenerate edges)."""
        obs = np.array([[5.0, 5.0]], dtype=np.float64)
        edges = list(edge_geometry_2d(self._SQUARE, obs))
        assert len(edges) == 4

    def test_valid_mask_shape(self) -> None:
        """valid mask has shape (M,) matching the number of observation points."""
        obs = np.array([[1.0, 1.0], [5.0, 5.0], [-3.0, -1.0]], dtype=np.float64)
        for _, _, dtheta, log_ratio, valid in edge_geometry_2d(self._SQUARE, obs):
            assert dtheta.shape == (3,)
            assert log_ratio.shape == (3,)
            assert valid.shape == (3,)


class TestEdgeGeometry25D:
    """Tests for edge_geometry_2_5d."""

    _RECT = np.array(
        [[-1.0, 1.0], [1.0, 1.0], [1.0, 3.0], [-1.0, 3.0]], dtype=np.float64
    )
    _OBS = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)

    def test_large_strike_dtheta_matches_2d(self) -> None:
        """dtheta from edge_geometry_2_5d with y0=1e7 ≈ dtheta from
        edge_geometry_2d."""
        y0 = 1e7
        edges_2d = list(edge_geometry_2d(self._RECT, self._OBS))
        edges_25d = list(edge_geometry_2_5d(self._RECT, self._OBS, y0))

        assert len(edges_2d) == len(edges_25d)
        for (
            (_, _, dt_2d, _, _),
            (_, _, dt_25d, _, _),
        ) in zip(edges_2d, edges_25d, strict=True):
            np.testing.assert_allclose(dt_25d, dt_2d, atol=1e-6)

    def test_large_strike_dlambda_matches_2d_log_ratio(self) -> None:
        """dlambda from edge_geometry_2_5d with y0=1e7 ≈ log_ratio
        from edge_geometry_2d."""
        y0 = 1e7
        edges_2d = list(edge_geometry_2d(self._RECT, self._OBS))
        edges_25d = list(edge_geometry_2_5d(self._RECT, self._OBS, y0))

        for (_, _, _, log_ratio, v_2d), (_, _, _, dlambda, v_25d) in zip(
            edges_2d, edges_25d, strict=True
        ):
            # Compare only at valid points (both masks should agree)
            mask = v_2d & v_25d
            np.testing.assert_allclose(dlambda[mask], log_ratio[mask], atol=1e-6)

    def test_won_bevis_theta_formula(self) -> None:
        """dtheta matches arctan2(z·y0, x·sqrt(r²+y0²)) formula directly."""
        # Triangle; first edge: vertex (1,1) → vertex (2,2)
        vertices = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float64)
        obs = np.array([[0.0, 0.0]], dtype=np.float64)
        y0 = 5.0

        first_edge = next(iter(edge_geometry_2_5d(vertices, obs, y0)))
        _, _, dtheta, _, valid = first_edge

        # Manual: vertex 0 = (1,1), vertex 1 = (2,2), obs = (0,0)
        x1, z1 = 1.0, 1.0
        x2, z2 = 2.0, 2.0
        r1 = np.sqrt(x1**2 + z1**2)
        r2 = np.sqrt(x2**2 + z2**2)
        sr1 = np.sqrt(r1**2 + y0**2)
        sr2 = np.sqrt(r2**2 + y0**2)
        theta1_exp = np.arctan2(z1 * y0, x1 * sr1)
        theta2_exp = np.arctan2(z2 * y0, x2 * sr2)
        dtheta_exp = theta2_exp - theta1_exp

        assert valid[0]
        assert np.isclose(dtheta[0], dtheta_exp, atol=1e-12), (
            f"dtheta {dtheta[0]:.12f} != expected {dtheta_exp:.12f}"
        )

    def test_won_bevis_lambda_formula(self) -> None:
        """dlambda matches log(r/(sqrt(r²+y0²)+y0)) formula directly."""
        # Triangle; first edge: vertex (1,1) → vertex (2,2), obs at origin
        vertices = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float64)
        obs = np.array([[0.0, 0.0]], dtype=np.float64)
        y0 = 5.0

        first_edge = next(iter(edge_geometry_2_5d(vertices, obs, y0)))
        _, _, _, dlambda, valid = first_edge

        # Manual: vertex 0 = (1,1), vertex 1 = (2,2), obs = (0,0)
        x1, z1 = 1.0, 1.0
        x2, z2 = 2.0, 2.0
        r1 = np.sqrt(x1**2 + z1**2)
        r2 = np.sqrt(x2**2 + z2**2)
        sr1 = np.sqrt(r1**2 + y0**2)
        sr2 = np.sqrt(r2**2 + y0**2)
        dlambda_exp = np.log(r2 / (sr2 + y0)) - np.log(r1 / (sr1 + y0))

        assert valid[0]
        assert np.isclose(dlambda[0], dlambda_exp, atol=1e-12), (
            f"dlambda {dlambda[0]:.12f} != expected {dlambda_exp:.12f}"
        )

    def test_degenerate_edge_skipped(self) -> None:
        """A zero-length edge is skipped just as in the 2D version."""
        vertices = np.array(
            [[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 3.0], [0.0, 3.0]],
            dtype=np.float64,
        )
        obs = np.array([[0.5, 0.0]], dtype=np.float64)
        edges = list(edge_geometry_2_5d(vertices, obs, 100.0))
        assert len(edges) == 4


class TestEdgeGeometry275D:
    """Tests for edge_geometry_2_75d."""

    _RECT = np.array(
        [[-1.0, 1.0], [1.0, 1.0], [1.0, 3.0], [-1.0, 3.0]], dtype=np.float64
    )
    _OBS = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)

    def test_symmetric_matches_25d(self) -> None:
        """When strike_forward == strike_backward == y0, yields equal 2.5D values."""
        y0 = 500.0
        edges_25d = list(edge_geometry_2_5d(self._RECT, self._OBS, y0))
        edges_275d = list(edge_geometry_2_75d(self._RECT, self._OBS, y0, y0))

        assert len(edges_25d) == len(edges_275d)
        for (_, _, dt_25d, dl_25d, _), (_, _, dt_275d, dl_275d, _) in zip(
            edges_25d, edges_275d, strict=True
        ):
            np.testing.assert_allclose(dt_275d, dt_25d, atol=1e-12)
            np.testing.assert_allclose(dl_275d, dl_25d, atol=1e-12)

    def test_asymmetric_is_per_edge_average(self) -> None:
        """Each yielded (dtheta, dlambda) equals the per-edge average
        of forward and backward 2.5D."""
        y_f = 700.0
        y_b = 200.0
        edges_fwd = list(edge_geometry_2_5d(self._RECT, self._OBS, y_f))
        edges_bwd = list(edge_geometry_2_5d(self._RECT, self._OBS, y_b))
        edges_275d = list(edge_geometry_2_75d(self._RECT, self._OBS, y_f, y_b))

        assert len(edges_fwd) == len(edges_275d)
        for (_, _, dt_f, dl_f, _), (_, _, dt_b, dl_b, _), (_, _, dt, dl, _) in zip(
            edges_fwd, edges_bwd, edges_275d, strict=True
        ):
            np.testing.assert_allclose(dt, (dt_f + dt_b) / 2.0, atol=1e-12)
            np.testing.assert_allclose(dl, (dl_f + dl_b) / 2.0, atol=1e-12)

    def test_tangent_vector_matches_forward(self) -> None:
        """tx, tz and valid from edge_geometry_2_75d match the forward 2.5D values."""
        y_f, y_b = 300.0, 100.0
        edges_fwd = list(edge_geometry_2_5d(self._RECT, self._OBS, y_f))
        edges_275d = list(edge_geometry_2_75d(self._RECT, self._OBS, y_f, y_b))

        for (tx_f, tz_f, _, _, v_f), (tx_275d, tz_275d, _, _, v_275d) in zip(
            edges_fwd, edges_275d, strict=True
        ):
            assert np.isclose(tx_275d, tx_f, atol=1e-15)
            assert np.isclose(tz_275d, tz_f, atol=1e-15)
            np.testing.assert_array_equal(v_275d, v_f)

    def test_yields_correct_edge_count(self) -> None:
        """Edge count equals that of the constituent 2.5D generators."""
        edges = list(edge_geometry_2_75d(self._RECT, self._OBS, 500.0, 200.0))
        assert len(edges) == 4
