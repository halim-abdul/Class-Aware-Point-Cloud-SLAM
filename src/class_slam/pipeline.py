import numpy as np
from .cloud import Cloud
from .registration import icp
from .graph import Edge, optimize

class SLAM:
    """Offline scan-to-scan odometry with verified proximity loop candidates."""
    def __init__(self, voxel_size=0.15, max_distance=0.8, dynamic=(10, 11),
                 min_inliers=20, max_rmse=0.3, loop_radius=1.0, loop_gap=10):
        self.voxel_size, self.max_distance, self.dynamic = voxel_size, max_distance, dynamic
        self.min_inliers, self.max_rmse = min_inliers, max_rmse
        self.loop_radius, self.loop_gap = loop_radius, loop_gap
        if voxel_size <= 0 or max_distance <= 0 or min_inliers < 3 or max_rmse <= 0 or loop_radius <= 0 or loop_gap < 2:
            raise ValueError("Invalid SLAM configuration")
        self.scans, self.poses, self.edges = [], [], []

    def accepted(self, result):
        return result.converged and result.inliers >= self.min_inliers and result.rmse <= self.max_rmse

    def add(self, cloud):
        scan = cloud.static(self.dynamic).voxel(self.voxel_size)
        if len(scan.points) < self.min_inliers:
            raise ValueError("Insufficient static points")
        if not self.scans:
            self.scans.append(scan); self.poses.append(np.eye(4))
            return self.poses[-1].copy()
        result = icp(scan, self.scans[-1], max_distance=self.max_distance)
        if not self.accepted(result):
            raise RuntimeError("Registration rejected; state unchanged")
        j = len(self.scans)
        pose = self.poses[-1] @ result.transform
        self.edges.append(Edge(j-1, j, result.transform))
        self.scans.append(scan); self.poses.append(pose)
        for i in range(max(0, j-self.loop_gap+1)):
            if np.linalg.norm(self.poses[i][:3,3] - pose[:3,3]) > self.loop_radius:
                continue
            initial = np.linalg.inv(self.poses[i]) @ pose
            loop = icp(scan, self.scans[i], initial=initial, max_distance=self.max_distance)
            if self.accepted(loop):
                self.edges.append(Edge(i, j, loop.transform))
                break
        return pose.copy()

    def optimize(self):
        self.poses = optimize(self.poses, self.edges)
        return self.poses

    def map(self):
        if not self.scans:
            return Cloud(np.empty((0,3)), np.empty(0, int), np.empty(0))
        clouds = [s.transform(p) for s,p in zip(self.scans,self.poses)]
        return Cloud(np.concatenate([c.points for c in clouds]),
                     np.concatenate([c.labels for c in clouds]),
                     np.concatenate([c.confidence for c in clouds])).voxel(self.voxel_size)
