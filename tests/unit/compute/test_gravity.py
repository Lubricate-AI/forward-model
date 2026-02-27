"""Tests for gravity anomaly computation (Talwani 1959)."""

import numpy as np

from forward_model.compute import GravityComponents, calculate_gravity
from forward_model.models import GeologicBody, GravityModel
from forward_model.models.properties import GravityProperties


class TestGravityKernel:
    """Basic sanity tests for gravity anomaly kernel."""

    def test_simple_rectangle_nonzero_output(self) -> None:
        """Test that a rectangle with density contrast produces non-zero anomaly."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=[25.0, 50.0, 0.0],
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        assert np.any(result.gz != 0.0)

    def test_output_shape_matches_observation_points(self) -> None:
        """Test that output shape matches number of observation points."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = list(np.linspace(-100.0, 100.0, 21))

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        assert result.gz.shape == (21,)
        assert result.gz_gradient.shape == (21,)

    def test_all_finite_values(self) -> None:
        """Test that all computed values are finite."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = list(np.linspace(-100.0, 100.0, 21))

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        assert np.all(np.isfinite(result.gz))
        assert np.all(np.isfinite(result.gz_gradient))

    def test_symmetric_polygon_symmetric_anomaly(self) -> None:
        """Test that symmetric polygon produces symmetric anomaly."""
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = [-50.0, -25.0, 0.0, 25.0, 50.0]

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        # Check symmetry: gz(-50) ≈ gz(50), etc.
        np.testing.assert_allclose(result.gz[0], result.gz[4], rtol=1e-10)
        np.testing.assert_allclose(result.gz[1], result.gz[3], rtol=1e-10)


class TestInfiniteSlab:
    """Slab geometry test: verify reasonable magnitudes."""

    def test_thin_horizontal_slab_produces_positive_anomaly(self) -> None:
        """Test gravity anomaly from thin horizontal slab is non-zero.

        A dense slab above the observation point should produce a measurable
        positive gravity anomaly (or negative if density contrast is negative).
        """
        # Create a thin, wide slab
        slab_thickness = 10.0  # meters
        slab_half_width = 5000.0  # 10 km wide

        vertices = np.array(
            [
                [-slab_half_width, 100.0],
                [slab_half_width, 100.0],
                [slab_half_width, 100.0 + slab_thickness],
                [-slab_half_width, 100.0 + slab_thickness],
            ],
            dtype=np.float64,
        )

        # Observation at surface
        obs_x = [0.0]

        density_contrast = 100.0  # kg/m³
        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=density_contrast),
                    name="Slab",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        computed_gz = result.gz[0]

        # Should produce a non-zero anomaly (magnitude > 0.001 mGal)
        assert np.abs(computed_gz) > 0.001, (
            f"Expected non-zero anomaly, got {computed_gz:.6e} mGal"
        )
        # For a positive density contrast above, anomaly should be positive
        assert computed_gz > 0.0, (
            f"Expected (+) anomaly for (+) density contrast, got {computed_gz:.6f} mGal"
        )


class TestGravityComponents:
    """Test GravityComponents dataclass."""

    def test_gravity_components_fields_exist(self) -> None:
        """Test that GravityComponents has expected fields."""
        gz = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        gz_gradient = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        comp = GravityComponents(gz=gz, gz_gradient=gz_gradient)
        assert comp.gz is gz
        assert comp.gz_gradient is gz_gradient

    def test_gravity_components_dtype(self) -> None:
        """Test that GravityComponents uses float64."""
        gz = np.array([1.0, 2.0], dtype=np.float64)
        gz_grad = np.array([0.1, 0.2], dtype=np.float64)
        comp = GravityComponents(gz=gz, gz_gradient=gz_grad)
        assert comp.gz.dtype == np.float64
        assert comp.gz_gradient.dtype == np.float64


class TestCalculateGravity:
    """Integration tests for calculate_gravity function."""

    def test_calculate_gravity_single_body(self) -> None:
        """Test gravity calculation with a single body."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = list(np.linspace(-100.0, 100.0, 11))

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        assert isinstance(result, GravityComponents)
        assert result.gz.shape == (11,)
        assert result.gz_gradient.shape == (11,)

    def test_calculate_gravity_multiple_bodies(self) -> None:
        """Test gravity calculation with multiple bodies via superposition."""
        # Two adjacent blocks
        verts1 = [[0.0, 100.0], [25.0, 100.0], [25.0, 200.0], [0.0, 200.0]]
        verts2 = [[25.0, 100.0], [50.0, 100.0], [50.0, 200.0], [25.0, 200.0]]
        obs_x = [12.5, 37.5]

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=verts1,
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block1",
                ),
                GeologicBody(
                    vertices=verts2,
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block2",
                ),
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        assert result.gz.shape == (2,)
        # Both points should have the same anomaly magnitude (symmetric blocks)
        assert np.isclose(np.abs(result.gz[0]), np.abs(result.gz[1]))

    def test_missing_gravity_properties_raises_error(self) -> None:
        """Test that missing gravity properties raises ValueError."""
        import pytest

        from forward_model.models.properties import MagneticProperties

        vertices = [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]]

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices,
                    magnetic=MagneticProperties(susceptibility=0.05),
                    name="MagneticOnly",
                )
            ],
            observation_x=[25.0],
            observation_z=0.0,
        )

        with pytest.raises(ValueError):
            calculate_gravity(model)

    def test_parallel_computation_matches_serial(self) -> None:
        """Test that parallel=True produces same result as serial."""
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = list(np.linspace(-100.0, 100.0, 21))

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result_serial = calculate_gravity(model, parallel=False)
        result_parallel = calculate_gravity(model, parallel=True)

        np.testing.assert_allclose(result_serial.gz, result_parallel.gz, rtol=1e-10)
        np.testing.assert_allclose(
            result_serial.gz_gradient, result_parallel.gz_gradient, rtol=1e-10
        )


class TestGravityStrikeDispatch:
    """Test 2.5D and 2.75D gravity computations."""

    def test_2_5d_finite_strike_attenuates_anomaly(self) -> None:
        """Test that 2.5D produces lower amplitude than 2D for same body."""
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = list(np.linspace(-100.0, 100.0, 21))

        # 2D model
        model_2d = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        # 2.5D model with finite strike
        model_25d = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                    strike_half_length=150.0,  # Finite strike
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result_2d = calculate_gravity(model_2d)
        result_25d = calculate_gravity(model_25d)

        # Peak 2.5D should be lower
        peak_2d = float(np.max(np.abs(result_2d.gz)))
        peak_25d = float(np.max(np.abs(result_25d.gz)))
        assert peak_25d < peak_2d, (
            f"Expected 2.5D peak ({peak_25d:.4f}) < 2D peak ({peak_2d:.4f})"
        )

    def test_2_75d_asymmetric_strike_computes(self) -> None:
        """Test that 2.75D with asymmetric strike produces finite values."""
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = [0.0, 25.0, 50.0]

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                    strike_forward=500.0,
                    strike_backward=200.0,
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        assert np.all(np.isfinite(result.gz))
        assert np.all(np.isfinite(result.gz_gradient))


class TestGravityNegativeContrast:
    """Test that negative density contrasts work correctly."""

    def test_negative_density_contrast_produces_negative_anomaly(self) -> None:
        """Test that negative density produces inverted anomaly."""
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = [0.0]

        # Positive contrast
        model_pos = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        # Negative contrast
        model_neg = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=-100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result_pos = calculate_gravity(model_pos)
        result_neg = calculate_gravity(model_neg)

        # Results should have opposite signs
        assert result_pos.gz[0] * result_neg.gz[0] < 0, (
            f"Expected opposite signs: pos={result_pos.gz[0]}, neg={result_neg.gz[0]}"
        )
        # Magnitudes should be equal
        assert np.isclose(np.abs(result_pos.gz[0]), np.abs(result_neg.gz[0]))


class TestGravityZeroContrast:
    """Test behavior with zero density contrast."""

    def test_zero_density_contrast_gives_zero_anomaly(self) -> None:
        """Test that zero density contrast gives zero anomaly."""
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = list(np.linspace(-100.0, 100.0, 21))

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=0.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        np.testing.assert_allclose(result.gz, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.gz_gradient, 0.0, atol=1e-12)


class TestGravityFarFieldDecay:
    """Test that anomaly decays with distance."""

    def test_anomaly_decreases_with_distance(self) -> None:
        """Test that gravity anomaly decreases at increasing distances."""
        vertices = np.array(
            [[0.0, 100.0], [50.0, 100.0], [50.0, 200.0], [0.0, 200.0]],
            dtype=np.float64,
        )
        # Observation points at increasing distances from body center
        obs_x = [25.0, 125.0, 500.0, 1000.0]

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)

        # Anomaly should decay monotonically with distance
        assert np.abs(result.gz[0]) > np.abs(result.gz[1])
        assert np.abs(result.gz[1]) > np.abs(result.gz[2])
        assert np.abs(result.gz[2]) > np.abs(result.gz[3])


class TestGravityGradient:
    """Test gradient computation."""

    def test_gradient_is_finite(self) -> None:
        """Test that computed gradient is finite."""
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        obs_x = list(np.linspace(-100.0, 100.0, 21))

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)
        assert np.all(np.isfinite(result.gz_gradient))

    def test_gradient_antisymmetric_for_symmetric_body(self) -> None:
        """Test that gradient is antisymmetric for symmetric body."""
        vertices = np.array(
            [[-25.0, 100.0], [25.0, 100.0], [25.0, 200.0], [-25.0, 200.0]],
            dtype=np.float64,
        )
        # Symmetric observation points around x=0
        obs_x = [-100.0, -50.0, 0.0, 50.0, 100.0]

        model = GravityModel(
            bodies=[
                GeologicBody(
                    vertices=vertices.tolist(),
                    gravity=GravityProperties(density_contrast=100.0),
                    name="Block",
                )
            ],
            observation_x=obs_x,
            observation_z=0.0,
        )

        result = calculate_gravity(model)

        # Gradient should be antisymmetric: grad(-x) == -grad(x)
        np.testing.assert_allclose(
            result.gz_gradient[0], -result.gz_gradient[4], atol=1e-6
        )
        np.testing.assert_allclose(
            result.gz_gradient[1], -result.gz_gradient[3], atol=1e-6
        )
        # At x=0, gradient should be ~0 (peak of symmetric anomaly)
        assert np.abs(result.gz_gradient[2]) < 1e-6
