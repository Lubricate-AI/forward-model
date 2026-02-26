"""Forward model container."""

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator

from forward_model.models.body import GeologicBody
from forward_model.models.field import MagneticField

_EPS = 1e-9


def _cross_2d(o: list[float], a: list[float], b: list[float]) -> float:
    """Return the 2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(
    p1: list[float],
    p2: list[float],
    p3: list[float],
    p4: list[float],
) -> bool:
    """Return True only for a proper (interior) crossing of two segments.

    Segments sharing an endpoint, touching at a corner, or collinear
    are all treated as non-intersecting.
    """
    d1 = _cross_2d(p3, p4, p1)
    d2 = _cross_2d(p3, p4, p2)
    d3 = _cross_2d(p1, p2, p3)
    d4 = _cross_2d(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    """Return True if point is strictly inside polygon (ray-casting).

    Points on the boundary are treated as outside.
    """
    x, z = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
            inside = not inside
        j = i
    return inside


def _point_on_segment(
    point: list[float], seg_a: list[float], seg_b: list[float]
) -> bool:
    """Return True if point lies on segment seg_a--seg_b (including endpoints)."""
    cross = (point[0] - seg_a[0]) * (seg_b[1] - seg_a[1]) - (point[1] - seg_a[1]) * (
        seg_b[0] - seg_a[0]
    )
    seg_len_sq = (seg_b[0] - seg_a[0]) ** 2 + (seg_b[1] - seg_a[1]) ** 2
    if cross * cross > seg_len_sq * _EPS * _EPS:
        return False
    # Check bounding-box containment
    return (
        min(seg_a[0], seg_b[0]) - _EPS <= point[0] <= max(seg_a[0], seg_b[0]) + _EPS
        and min(seg_a[1], seg_b[1]) - _EPS <= point[1] <= max(seg_a[1], seg_b[1]) + _EPS
    )


def _point_on_polygon_boundary(point: list[float], polygon: list[list[float]]) -> bool:
    """Return True if point lies on any edge of polygon."""
    n = len(polygon)
    for i in range(n):
        if _point_on_segment(point, polygon[i], polygon[(i + 1) % n]):
            return True
    return False


def _polygon_centroid(polygon: list[list[float]]) -> list[float]:
    """Return the area-weighted centroid [x, z] via the shoelace formula.

    Falls back to vertex mean for degenerate (zero-area) polygons.
    """
    n = len(polygon)
    area = 0.0
    cx = 0.0
    cz = 0.0
    for i in range(n):
        x0, z0 = polygon[i]
        x1, z1 = polygon[(i + 1) % n]
        cross = x0 * z1 - x1 * z0
        area += cross
        cx += (x0 + x1) * cross
        cz += (z0 + z1) * cross
    area /= 2.0
    if abs(area) < 1e-15:
        return [sum(v[0] for v in polygon) / n, sum(v[1] for v in polygon) / n]
    cx /= 6.0 * area
    cz /= 6.0 * area
    return [cx, cz]


def _polygons_overlap(poly1: list[list[float]], poly2: list[list[float]]) -> bool:
    """Return True if poly1 and poly2 have a non-trivial overlap.

    Returns False for polygons that only share an edge or corner.
    """
    n1, n2 = len(poly1), len(poly2)

    for i in range(n1):
        p1, p2 = poly1[i], poly1[(i + 1) % n1]
        for j in range(n2):
            p3, p4 = poly2[j], poly2[(j + 1) % n2]
            if _segments_intersect(p1, p2, p3, p4):
                return True

    for vertex in poly1:
        if not _point_on_polygon_boundary(vertex, poly2) and _point_in_polygon(
            vertex, poly2
        ):
            return True

    for vertex in poly2:
        if not _point_on_polygon_boundary(vertex, poly1) and _point_in_polygon(
            vertex, poly1
        ):
            return True

    # Check centroid containment to catch coincident polygons and collinear-edge
    # overlaps where all vertices land on the other polygon's boundary.
    c1 = _polygon_centroid(poly1)
    if _point_in_polygon(c1, poly2):
        return True
    c2 = _polygon_centroid(poly2)
    if _point_in_polygon(c2, poly1):
        return True

    return False


class ForwardModel(BaseModel, frozen=True):
    """Complete forward magnetic model specification.

    Attributes:
        bodies: List of geologic bodies to include in the model.
                Must contain at least one body.
        field: Earth's magnetic field parameters.
        observation_x: List of x-coordinates for observation points (meters).
        observation_z: Fixed z-coordinate for all observation points (meters).
                      Typically 0 for surface observations.
    """

    bodies: list[GeologicBody] = Field(min_length=1)
    field: MagneticField
    observation_x: list[float]
    observation_z: float

    @field_validator("observation_x")
    @classmethod
    def validate_observation_x(cls, v: list[float]) -> list[float]:
        """Validate observation x coordinates are finite."""
        if not all(math.isfinite(x) for x in v):
            raise ValueError("All observation x coordinates must be finite")
        return v

    @field_validator("observation_z")
    @classmethod
    def validate_observation_z(cls, v: float) -> float:
        """Validate observation z coordinate is finite."""
        if not math.isfinite(v):
            raise ValueError(f"observation_z must be finite, got {v}")
        return v

    @model_validator(mode="after")
    def validate_no_body_overlap(self) -> "ForwardModel":
        """Validate that no two geologic bodies overlap.

        Bodies sharing only an edge or corner are permitted.
        """
        bodies = self.bodies
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                a, b = bodies[i], bodies[j]
                if _polygons_overlap(a.vertices, b.vertices):
                    raise ValueError(f"Bodies '{a.name}' and '{b.name}' overlap")
        return self

    def get_observation_points(self) -> NDArray[np.float64]:
        """Get observation points as NumPy array.

        Returns:
            Nx2 array where N is the number of observation points.
            Each row is [x, z] coordinates.
        """
        n_points = len(self.observation_x)
        points = np.zeros((n_points, 2), dtype=np.float64)
        points[:, 0] = self.observation_x
        points[:, 1] = self.observation_z
        return points
