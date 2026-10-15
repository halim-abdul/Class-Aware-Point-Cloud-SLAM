# Data

Coordinates: metres, right-handed sensor frame; timestamps and synchronisation are handled upstream.
NPZ arrays are points (N,3), labels (N,) integer, confidence (N,) in [0,1].
Non-finite points and invalid confidence are rejected.

Demo taxonomy: 0 = lower static landmarks, 1 = upper static landmarks, 10 = moving objects.
Default dynamic IDs are (10,11); adapt these to your dataset. Unknown camera projections use label -1 and zero confidence and are removed by the confidence gate.

Voxel filtering preserves classes and keeps the highest-confidence observation in each class/voxel pair. It does not average mixed semantic categories.
Real datasets must be downloaded separately under their own licences. No dataset is bundled.
