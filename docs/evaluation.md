# Evaluation

Run pytest -q and class-slam --seed 7 --output outputs/demo.
The synthetic smoke test checks translation ATE < 0.03 m and at least one non-adjacent loop.
This threshold is a regression criterion for a shared-view toy sequence, not a real-world accuracy claim.

ATE is translation RMSE in a shared coordinate frame without automatic alignment.
RPE compares successive relative transforms and reports translation RMSE.
Rotation error, runtime, memory, map completeness and dynamic leakage are not yet included in the CLI report.

For a real experiment:
1. Select fixed train/validation/test sequences and publish preprocessing and calibration.
2. Compare geometry-only ICP (semantic=False), semantic matching, and dynamic filtering separately.
3. Sweep class confidence, voxel size, overlap and moving-object fraction with several seeds.
4. Report failures as well as successful trajectories, trajectory alignment method, and timing hardware.
5. Evaluate noisy/missing labels and incorrect class IDs; supplied ground-truth labels overestimate practical semantic performance.

Never report synthetic metrics as KITTI, SemanticKITTI or robot results.
