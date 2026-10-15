import argparse
import json
from pathlib import Path
import numpy as np
from .cloud import Cloud
from .pipeline import SLAM
from .metrics import translation_ate, translation_rpe

def sequence(seed=7, frames=24):
    """Synthetic shared-view 3D landmarks with independently moving objects."""
    rng = np.random.default_rng(seed)
    world = rng.uniform([-5,-5,-1],[5,5,3],(600,3))
    labels = (world[:,2]>1).astype(int)
    for k in range(frames):
        theta = 2*np.pi*k/(frames-1)
        pose = np.eye(4)
        pose[:3,3] = [0.6*np.sin(theta),0.6*(1-np.cos(theta)),0.03*np.sin(theta)]
        static = world-pose[:3,3] + rng.normal(0,0.003,world.shape)
        moving = rng.normal([2*np.cos(theta),1,0.5],0.3,(80,3))
        yield Cloud(np.vstack([static,moving]), np.r_[labels,np.full(80,10)], np.ones(680)), pose

def run(output, seed=7):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    slam = SLAM(loop_gap=12)
    truth=[]
    for scan,pose in sequence(seed):
        slam.add(scan); truth.append(pose)
    before = translation_ate(slam.poses,truth)
    slam.optimize()
    slam.map().save(output/"map.npz")
    np.save(output/"poses.npy",np.array(slam.poses))
    np.save(output/"truth.npy",np.array(truth))
    report = dict(dataset="synthetic shared-view landmarks",seed=seed,frames=len(truth),
                  ate_before_m=before,ate_after_m=translation_ate(slam.poses,truth),
                  rpe_m=translation_rpe(slam.poses,truth),
                  loop_edges=sum(e.j-e.i>1 for e in slam.edges),
                  map_points=len(slam.map().points))
    (output/"metrics.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="outputs/demo")
    p.add_argument("--seed",type=int,default=7)
    args=p.parse_args()
    run(args.output,args.seed)

if __name__=="__main__":
    main()
