from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

@dataclass
class Registration:
    transform: np.ndarray
    rmse: float
    inliers: int
    converged: bool
    iterations: int

def rigid_fit(source, target, weights):
    """Weighted Kabsch: target ~= R source + t; reflection is forbidden."""
    w = np.asarray(weights, float)
    if len(source) < 3 or w.sum() <= 0:
        raise ValueError("At least three positive-weight correspondences required")
    w = w / w.sum()
    a, b = w @ source, w @ target
    H = (source - a).T @ ((target - b) * w[:, None])
    U, s, Vt = np.linalg.svd(H)
    if s[1] < 1e-12:
        raise ValueError("Collinear or coincident correspondences")
    D = np.eye(3)
    D[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, b - R @ a
    return T

def icp(source, target, initial=None, max_distance=1.0, max_iterations=50,
        tolerance=1e-6, semantic=True, robust_scale=0.2):
    """Source-to-target SE(3) ICP with same-class matches and Huber weights."""
    if max_distance <= 0 or robust_scale <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("Invalid ICP settings")
    T = np.eye(4) if initial is None else np.asarray(initial, float).copy()
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Initial transform must be finite 4x4")
    groups = {}
    for label in (np.unique(target.labels) if semantic else [None]):
        ids = np.flatnonzero(target.labels == label) if semantic else np.arange(len(target.points))
        if len(ids):
            groups[label] = (ids, cKDTree(target.points[ids]))
    def matches(points):
        si, ti, distances = [], [], []
        for label, (ids, tree) in groups.items():
            rows = np.flatnonzero(source.labels == label) if semantic else np.arange(len(points))
            if not len(rows):
                continue
            d, j = tree.query(points[rows], distance_upper_bound=max_distance)
            valid = np.isfinite(d)
            si.extend(rows[valid]); ti.extend(ids[j[valid]]); distances.extend(d[valid])
        return np.array(si, int), np.array(ti, int), np.array(distances)
    converged = False
    for iteration in range(1, max_iterations + 1):
        moved = source.points @ T[:3, :3].T + T[:3, 3]
        si, ti, d = matches(moved)
        w = source.confidence[si] * target.confidence[ti]
        w *= np.minimum(1., robust_scale / np.maximum(d, 1e-12))
        valid = w > 0
        if valid.sum() < 3:
            return Registration(T, float("inf"), int(valid.sum()), False, iteration)
        try:
            delta = rigid_fit(moved[si[valid]], target.points[ti[valid]], w[valid])
        except ValueError:
            return Registration(T, float("inf"), int(valid.sum()), False, iteration)
        T = delta @ T
        if np.linalg.norm(delta - np.eye(4)) < tolerance:
            converged = True
            break
    moved = source.points @ T[:3, :3].T + T[:3, 3]
    si, ti, d = matches(moved)
    valid = source.confidence[si] * target.confidence[ti] > 0
    return Registration(T, float(np.sqrt(np.mean(d[valid] ** 2))) if valid.any() else float("inf"),
                        int(valid.sum()), converged, iteration)
