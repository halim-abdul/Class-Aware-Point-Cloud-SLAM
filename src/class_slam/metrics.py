import numpy as np

def translation_ate(estimated, reference):
    """Translation RMSE in a shared world frame; no implicit alignment."""
    a,b = np.asarray(estimated),np.asarray(reference)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (4,4) or not len(a):
        raise ValueError("Expected matching nonempty (N,4,4) pose arrays")
    return float(np.sqrt(np.mean(np.sum((a[:,:3,3]-b[:,:3,3])**2,axis=1))))

def translation_rpe(estimated, reference):
    """One-frame relative translation RMSE."""
    a,b = np.asarray(estimated),np.asarray(reference)
    translation_ate(a,b)
    if len(a)<2:
        raise ValueError("At least two poses required")
    errors = [np.linalg.inv(np.linalg.inv(b[i])@b[i+1]) @
              (np.linalg.inv(a[i])@a[i+1]) for i in range(len(a)-1)]
    return float(np.sqrt(np.mean([np.sum(e[:3,3]**2) for e in errors])))
