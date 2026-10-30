import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from common import unit
from geometry import expected_composition,pair_accuracy,Neighbours

def test_float16_norm_is_safe():
 x=np.full((2,1000),100,np.float16);np.testing.assert_allclose(np.linalg.norm(unit(x),axis=1),1,atol=1e-6)
def test_neighbours_exclude_self_and_obey_source():
 x=unit(np.array([[1,0],[.9,.1],[0,1],[-1,0]],float));n=Neighbours(x);ii,_=n.get([0],[1,2,3],2);assert ii.tolist()==[[1,2]]
 ii,_=n.get([0],[0,1,2],2);assert 0 not in ii[0]
def test_random_control_matches_enumeration():
 y=np.array([[0.],[2.],[4.],[3.]]);groups=np.array(['a','a','b','q']);nb=np.array([[0,2]]);tr=np.array([0,1,2]);te=np.array([3]);pair,mean,mu=expected_composition(y,te,tr,nb,groups)
 pairs=np.array([((y[a,0]-3)**2+(4-3)**2)/2 for a in [0,1]])
 errors=np.array([((y[a,0]+4)/2-3)**2 for a in [0,1]])
 np.testing.assert_allclose(pair,pairs.mean());np.testing.assert_allclose(mean,errors.mean());np.testing.assert_allclose(mu,[[2.5]])
def test_pairwise_ties_not_inflated():
 a=np.array([[1,1,0],[0,1,2]],float);b=np.array([[2,1,0],[1,1,1]],float);score,n=pair_accuracy(a,b)
 assert np.isclose(score[0],2.5/3);assert n[1]==0 and np.isnan(score[1])

def test_small_predicted_difference_gets_only_half_credit():
 a=np.array([[1e-14,0.]],float);b=np.array([[1.,0.]],float);score,n=pair_accuracy(a,b)
 assert n[0]==1 and score[0]==.5

def test_small_source_caps_k_after_excluding_self():
 x=unit(np.array([[1,0],[.9,.1],[0,1],[-1,0]],float))
 n=Neighbours(x)
 ii,cc=n.get([0,3],[0,1,2],100)
 assert ii.shape==(2,2) and np.isfinite(cc).all()
 assert 0 not in ii[0] and np.isin(ii,[0,1,2]).all()
def test_no_nonself_source_and_duplicates_are_rejected():
 import pytest
 n=Neighbours(unit(np.array([[1,0],[0,1]],float)))
 with pytest.raises(ValueError,match='No distinct'):n.get([0],[0],100)
 with pytest.raises(ValueError,match='distinct'):n.get([0],[1,1],100)
