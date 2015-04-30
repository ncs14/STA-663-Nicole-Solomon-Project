
from ffsr import bagfsr
from numpy.testing import assert_raises

import numpy as np
import pandas as pd

np.random.seed(1234)

g = np.random.random(15)

X = np.random.multivariate_normal(np.zeros(15),np.eye(15),(100))
beta = np.array([0,0,5,6,0,0,4,0,0,0,5,0,0,0,0]).reshape(15,1) # signif betas: 3,4,7,11
Y = X.dot(beta)
Y2 = pd.DataFrame(Y)
X2 = pd.DataFrame(X)
X2.columns = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15"]
d = pd.concat([Y2,X2],axis=1)

def test_toomanyp():
    assert_raises(ValueError,bagfsr,d.iloc[:10,],0.05)

def test_invalidg():
    assert_raises(ValueError,bagfsr,d,0)
    
def test_invalidB():
    assert_raises(ValueError,bagfsr,d,0.05,0)