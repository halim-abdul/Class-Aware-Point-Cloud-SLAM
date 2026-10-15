# Method and frame conventions

Use column-vector homogeneous transformations: p_world = T_world_sensor p_sensor.
Array implementation uses points @ R.T + t. ICP(source,target) returns T_target_source.
Odometry composes T_world_current = T_world_previous T_previous_current.

For each transformed source point, find the closest target point of the same semantic class within a distance gate.
Pair weights multiply source confidence, target confidence and min(1, robust_scale / distance).
Weighted Kabsch minimizes sum_i w_i ||R p_i + t - q_i||^2, with det(R)=+1.
Collinear/coincident matches are rejected. Planar non-collinear matches are allowed.
ICP accepts an initial guess but the pipeline currently starts adjacent-scan registration at identity.

Graph edge Z_ij maps scan j into scan i. The residual packs translation and rotation vector of:
inverse(Z_ij) inverse(T_world_i) T_world_j.
The first pose is fixed. A soft-L1 loss limits large residual influence.
This residual is a practical SE(3) parameterization, not the full Lie algebra logarithm with translation Jacobian.
Translation metres and rotation radians have equal numerical scale; edge weights are scalar, not estimated information matrices.

A loop requires temporal separation, spatial proximity and successful semantic ICP.
Convergence, minimum inliers and maximum RMSE all gate acceptance. Failed odometry does not modify state.
After optimization the map is rebuilt from retained sensor-frame scans so pose corrections reach every map point.
