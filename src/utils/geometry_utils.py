"""Geometry utility functions for the CAD Drawing Generator."""
import math
from typing import List, Tuple, Optional

import numpy as np


class GeometryUtils:
    """Collection of geometric helper methods."""

    @staticmethod
    def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Compute Euclidean distance between two 3-D points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    @staticmethod
    def normalize_vector(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Return the unit vector of *v*."""
        mag = math.sqrt(sum(x ** 2 for x in v))
        if mag < 1e-10:
            return (0.0, 0.0, 0.0)
        return tuple(x / mag for x in v)  # type: ignore[return-value]

    @staticmethod
    def cross_product(
        v1: Tuple[float, float, float], v2: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Return the cross product of two 3-D vectors."""
        return (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        )

    @staticmethod
    def dot_product(
        v1: Tuple[float, float, float], v2: Tuple[float, float, float]
    ) -> float:
        """Return the dot product of two 3-D vectors."""
        return sum(a * b for a, b in zip(v1, v2))

    @staticmethod
    def compute_bounding_box(
        points: List[Tuple[float, float, float]],
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Compute axis-aligned bounding box for a list of 3-D points.

        Returns:
            Tuple of (min_point, max_point).
        """
        if not points:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        arr = np.array(points, dtype=float)
        mn = tuple(arr.min(axis=0).tolist())
        mx = tuple(arr.max(axis=0).tolist())
        return mn, mx  # type: ignore[return-value]

    @staticmethod
    def compute_face_area(vertices: List[Tuple[float, float, float]]) -> float:
        """
        Estimate face area using the shoelace / cross-product method for a polygon.

        Works for convex or simple polygons projected in 3-D space.
        """
        if len(vertices) < 3:
            return 0.0
        arr = np.array(vertices, dtype=float)
        n = len(arr)
        area = 0.0
        origin = arr[0]
        for i in range(1, n - 1):
            v1 = arr[i] - origin
            v2 = arr[i + 1] - origin
            cross = np.cross(v1, v2)
            area += np.linalg.norm(cross)
        return area / 2.0

    @staticmethod
    def compute_face_normal(
        vertices: List[Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        """Compute an approximate face normal from the first three vertices."""
        if len(vertices) < 3:
            return (0.0, 0.0, 1.0)
        arr = np.array(vertices[:3], dtype=float)
        v1 = arr[1] - arr[0]
        v2 = arr[2] - arr[0]
        normal = np.cross(v1, v2)
        mag = np.linalg.norm(normal)
        if mag < 1e-10:
            return (0.0, 0.0, 1.0)
        return tuple((normal / mag).tolist())  # type: ignore[return-value]

    @staticmethod
    def angle_between_vectors(
        v1: Tuple[float, float, float], v2: Tuple[float, float, float]
    ) -> float:
        """Return the angle in radians between two 3-D vectors."""
        dot = GeometryUtils.dot_product(v1, v2)
        mag1 = math.sqrt(sum(x ** 2 for x in v1))
        mag2 = math.sqrt(sum(x ** 2 for x in v2))
        if mag1 < 1e-10 or mag2 < 1e-10:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.acos(cos_angle)

    @staticmethod
    def project_point_to_plane(
        point: Tuple[float, float, float],
        plane_normal: Tuple[float, float, float],
        plane_origin: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """
        Project a 3-D point onto a plane and return 2-D coordinates.

        Builds a local coordinate frame on the plane using Gram-Schmidt.
        """
        n = np.array(plane_normal, dtype=float)
        n /= np.linalg.norm(n) + 1e-12
        p = np.array(point, dtype=float)
        o = np.array(plane_origin, dtype=float)

        # Build tangent vectors
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - np.dot(ref, n) * n
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(n, u)

        delta = p - o
        return float(np.dot(delta, u)), float(np.dot(delta, v))

    @staticmethod
    def fit_circle(
        points: List[Tuple[float, float]],
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        Fit a circle to 2-D points using a least-squares algebraic method.

        Returns:
            ((cx, cy), radius) or None when fitting fails.
        """
        pts = np.array(points, dtype=float)
        if len(pts) < 3:
            return None
        A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
        b = pts[:, 0] ** 2 + pts[:, 1] ** 2
        try:
            result, *_ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy, c = result
            r = math.sqrt(c + cx ** 2 + cy ** 2)
            return (float(cx), float(cy)), float(r)
        except np.linalg.LinAlgError:
            return None
