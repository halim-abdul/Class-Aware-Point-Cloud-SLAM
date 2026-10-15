# ROS2 and C++ integration

The Python library runs without ROS. A live ROS2 node is not included or tested.

Suggested adapter contract:
- Subscribe to sensor_msgs/PointCloud2 plus a timestamp-aligned semantic image.
- Convert XYZ values to metres and reject invalid measurements upstream.
- Obtain camera_from_lidar extrinsics from tf2; use rectified camera intrinsics.
- Call camera_labels only after timestamp and calibration checks.
- Feed Cloud instances to SLAM.add from a single worker.
- Publish poses as nav_msgs/Odometry and the world-to-sensor transform through tf2.
- Publish the rebuilt global cloud after graph optimization.
- Maintain one world frame and avoid publishing competing TF authorities.

Projection currently performs no occlusion z-buffering or lens distortion correction.
An upstream adapter must provide rectified images and resolve occlusion when needed.
No ROS2 distribution compatibility or live latency claim is made.

The standalone C++17 example demonstrates semantic/confidence filtering only. It is not a C++ SLAM backend.
