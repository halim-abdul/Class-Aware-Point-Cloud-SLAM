"""Small dense SE(3) pose graph; first pose fixed to remove gauge freedom."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def pack(T):
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()]

def unpack(x):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[3:]).as_matrix()
    T[:3, 3] = x[:3]
    return T

@dataclass
class Edge:
    i: int
    j: int
    measurement: np.ndarray
    weight: float = 1.0

def optimize(poses, edges):
    """Edge Z_ij maps coordinates of scan j to scan i."""
    if len(poses) < 2:
        return [p.copy() for p in poses]
    for e in edges:
        if not (0 <= e.i < len(poses) and 0 <= e.j < len(poses)) or e.weight <= 0:
            raise ValueError("Invalid graph edge")
    def decode(x):
        return [poses[0]] + [unpack(row) for row in x.reshape(-1, 6)]
    def residual(x):
        ps = decode(x)
        return np.concatenate([np.sqrt(e.weight) * pack(
            np.linalg.inv(e.measurement) @ np.linalg.inv(ps[e.i]) @ ps[e.j]) for e in edges])
    if not edges:
        return [p.copy() for p in poses]
    x0 = np.concatenate([pack(p) for p in poses[1:]])
    result = least_squares(residual, x0, loss="soft_l1", max_nfev=200)
    if not result.success:
        raise RuntimeError("Pose graph optimization failed: " + result.message)
    return decode(result.x)
