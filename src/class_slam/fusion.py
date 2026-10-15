"""Project calibrated LiDAR points into an already-classified camera image."""
import numpy as np
from .cloud import Cloud

def camera_labels(points, image_labels, K, camera_from_lidar, confidence=None):
    image_labels = np.asarray(image_labels)
    if image_labels.ndim != 2 or not np.issubdtype(image_labels.dtype, np.integer):
        raise ValueError("Expected integer HxW semantic image")
    points = np.asarray(points, float)
    pc = points @ camera_from_lidar[:3,:3].T + camera_from_lidar[:3,3]
    uvw = pc @ np.asarray(K).T
    valid = pc[:,2] > 0
    uv = np.zeros((len(points),2), int)
    uv[valid] = np.floor(uvw[valid,:2] / uvw[valid,2,None]).astype(int)
    h,w = image_labels.shape
    valid &= (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    labels = np.full(len(points), -1, int)
    conf = np.zeros(len(points))
    labels[valid] = image_labels[uv[valid,1],uv[valid,0]]
    conf[valid] = 1. if confidence is None else np.asarray(confidence)[uv[valid,1],uv[valid,0]]
    return Cloud(points,labels,conf)
