# Limitations and research roadmap

Current scope is an offline, small-scale semantic SLAM baseline.
No trained camera classifier, ROS2 node, IMU fusion, deskewing, global relocalization, map aging or production deployment is included.

Priority research extensions:
- Motion-aware semantic policies: parked cars should not automatically equal moving cars.
- Semantic uncertainty calibration and unknown-class handling.
- Point-to-plane matching and multiscale coarse-to-fine initialization.
- Descriptor-based place recognition with independent geometric verification.
- Sparse pose-graph solvers with calibrated information matrices.
- Timestamped voxels with persistence, decay and free-space evidence.
- Public dataset adapters and real robot recordings with reproducible ablations.

Any camera/LiDAR system also needs explicit time synchronization, intrinsics/extrinsics validation and occlusion handling.
