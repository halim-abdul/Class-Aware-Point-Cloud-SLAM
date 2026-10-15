from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Cloud:
    """Points (N,3), integer semantic labels (N,), confidence (N,)."""
    points: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.points, dtype=float)
        raw = np.asarray(self.labels)
        c = np.asarray(self.confidence, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or raw.shape != (len(p),) or c.shape != (len(p),):
            raise ValueError("Expected points (N,3), labels and confidence (N,)")
        if not np.isfinite(p).all() or not np.isfinite(c).all():
            raise ValueError("Non-finite point or confidence")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Labels must be integer IDs")
        if ((c < 0) | (c > 1)).any():
            raise ValueError("Confidence must be in [0,1]")
        object.__setattr__(self, "points", p.copy())
        object.__setattr__(self, "labels", raw.astype(int, copy=True))
        object.__setattr__(self, "confidence", c.copy())

    def subset(self, mask):
        return Cloud(self.points[mask], self.labels[mask], self.confidence[mask])

    def static(self, dynamic=(10, 11), min_confidence=0.3):
        return self.subset(~np.isin(self.labels, dynamic) & (self.confidence >= min_confidence))

    def voxel(self, size=0.2):
        """Keep the most confident observation per spatial voxel and class."""
        if size <= 0:
            raise ValueError("Voxel size must be positive")
        keys = np.column_stack((np.floor(self.points / size).astype(np.int64), self.labels))
        order = np.argsort(-self.confidence, kind="stable")
        _, first = np.unique(keys[order], axis=0, return_index=True)
        return self.subset(np.sort(order[first]))

    def transform(self, T):
        return Cloud(self.points @ T[:3, :3].T + T[:3, 3], self.labels, self.confidence)

    def save(self, path):
        np.savez_compressed(path, points=self.points, labels=self.labels, confidence=self.confidence)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data["points"], data["labels"], data["confidence"])
