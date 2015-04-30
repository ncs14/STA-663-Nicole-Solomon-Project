
from ffsr import alpha_F_g
from numpy.testing import assert_raises, assert_almost_equal

import numpy as np

np.random.seed(1234)

g = np.random.random(15)
g2 = np.array([0.01,0.02,0.03])

def test_nogamma0():    
    assert_raises(ValueError,alpha_F_g,None,g,15)
    
def test_nogamma1():
    assert_raises(ValueError,alpha_F_g,0.05,None,15)
    
def test_nogamma2():
    assert_raises(ValueError,alpha_F_g,g2,None,15)
    
def test_g0toobig1():
    assert_raises(ValueError,alpha_F_g,1,g,15)
    
def test_g0toobig2():
    assert_raises(ValueError,alpha_F_g,np.array([0.05,1]),g,15)
    
def test_g0toosm1():
    assert_raises(ValueError,alpha_F_g,-0.1,g,15)
    
def test_g0toosm2():
    assert_raises(ValueError,alpha_F_g,np.array([-0.05,0.1]),g,15)
    
def test_g0toobgsm():
    assert_raises(ValueError,alpha_F_g,np.array([-0.05,0.1,1.]),g,15)
    
def test_noncov1():
    assert_raises(TypeError,alpha_F_g,0.05,g,None)
    
def test_noncov1():
    assert_raises(TypeError,alpha_F_g,g2,g,None)
    
def test_known():
    g3 = np.array([0.1,0.5])
    n = 5
    g0 = 0.25
    s = min(np.where(g3>g0)[0])
    a = 0.25*2/4. # g0 * (1. + s) / (n - s)
    afg = alpha_F_g(g0,g3,n)
    assert_almost_equal(a,afg)