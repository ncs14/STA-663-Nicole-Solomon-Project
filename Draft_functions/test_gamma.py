
from ffsr import gamma_F
from numpy.testing import assert_raises

import numpy as np

np.random.seed(1234)

po = np.random.random(15)
pm = np.array([max(po[:(i+1)]) for i in range(len(po))])
s = np.array([len(np.where(pm<=pm[i])[0]) for i in range(len(pm))])

def test_nopvals():    
    assert_raises(TypeError,gamma_F,None,15,s)
    
def test_stoobig():
    assert_raises(ValueError,gamma_F,pm[:10],10,s)
    
def test_ncovtoosmall():
    assert_raises(ValueError,gamma_F,pm,10,s)