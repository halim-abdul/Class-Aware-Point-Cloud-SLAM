import numpy as np
import pytest
from class_slam import Cloud, SLAM, icp
from class_slam.registration import rigid_fit
from class_slam.graph import Edge,optimize
from class_slam.fusion import camera_labels
from class_slam.metrics import translation_ate,translation_rpe
from class_slam.demo import sequence,run

def cloud(seed=3):
    p=np.random.default_rng(seed).normal(size=(100,3))
    return Cloud(p,np.zeros(100,int),np.ones(100))

def test_rigid_rotation_translation():
    a=cloud().points
    T=np.eye(4); T[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]; T[:3,3]=[2,-1,0.3]
    b=a@T[:3,:3].T+T[:3,3]
    np.testing.assert_allclose(rigid_fit(a,b,np.ones(100)),T,atol=1e-12)

def test_icp_direction():
    a=cloud(); T=np.eye(4); T[:3,3]=[0.1,-0.05,0.02]
    r=icp(a,a.transform(T))
    assert r.converged
    np.testing.assert_allclose(r.transform,T,atol=1e-6)

def test_semantic_mismatch():
    a=cloud(); b=Cloud(a.points,np.ones(100,int),a.confidence)
    assert icp(a,b).inliers==0
    assert icp(a,b,semantic=False).converged

def test_degenerate():
    a=np.zeros((4,3))
    with pytest.raises(ValueError): rigid_fit(a,a,np.ones(4))

def test_dynamic_and_voxel():
    a=Cloud([[0,0,0],[0.01,0,0],[1,0,0]],np.array([0,0,10]),[0.5,0.9,1])
    b=a.static().voxel(0.1)
    assert len(b.points)==1
    assert b.confidence[0]==0.9

def test_invalid_cloud():
    with pytest.raises(ValueError): Cloud([[np.nan,0,0]],np.array([0]),[1])
    with pytest.raises(ValueError): Cloud([[0,0,0]],np.array([0]),[2])

def test_io(tmp_path):
    a=cloud(); a.save(tmp_path/"a.npz")
    np.testing.assert_array_equal(Cloud.load(tmp_path/"a.npz").points,a.points)

def test_projection():
    c=camera_labels(np.array([[0,0,1],[0,0,-1],[20,0,1]]),
                    np.array([[4,5],[6,7]]),np.eye(3),np.eye(4))
    np.testing.assert_array_equal(c.labels,[4,-1,-1])

def test_rejected_scan_preserves_state():
    s=SLAM(); a=cloud(); s.add(a)
    T=np.eye(4); T[:3,3]=100
    with pytest.raises(RuntimeError): s.add(a.transform(T))
    assert len(s.poses)==len(s.scans)==1 and not s.edges

def test_graph_reduces_error():
    truth=[np.eye(4) for _ in range(3)]
    for i,p in enumerate(truth): p[0,3]=i
    poses=[p.copy() for p in truth]; poses[-1][0,3]+=0.3
    z=np.eye(4); z[0,3]=1
    result=optimize(poses,[Edge(0,1,z),Edge(1,2,z)])
    assert translation_ate(result,truth)<1e-6
    assert translation_rpe(result,truth)<1e-6
    np.testing.assert_array_equal(result[0],truth[0])

def test_demo(tmp_path):
    result=run(tmp_path)
    assert result["ate_after_m"]<0.03
    assert result["loop_edges"]>0
    assert (tmp_path/"map.npz").exists()
