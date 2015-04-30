
from ffsr import alpha_F
from numpy.testing import assert_raises, assert_almost_equal
import numpy as np

s = np.arange(1,15)

def test_nogamma0():    
    assert_raises(ValueError,alpha_F,None,15,s)
    
def test_noncov():
    assert_raises(ValueError,alpha_F,0.05,None,s)
    
def test_stoobig():
    assert_raises(ValueError,alpha_F,0.05,10,s)
    
def test_known():
    s = np.arange(1,5)
    n = 5
    a = 0.5 * (1 + s) / (n - s)
    af = alpha_F(0.5,n,np.arange(1,n))
    assert_almost_equal(a,af)