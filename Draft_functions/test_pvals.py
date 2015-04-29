
from ffsr import *
from numpy.testing import assert_raises

import rpy2.robjects as ro
import pandas.rpy.common as com

import numpy as np
import pandas as pd

np.random.seed(1234)

X = np.random.multivariate_normal(np.zeros(15),np.eye(15),(100))
beta = np.array([0,0,5,6,0,0,4,0,0,0,5,0,0,0,0]).reshape(15,1) # signif betas: 3,4,7,11
Y = X.dot(beta)
Y2 = pd.DataFrame(Y)
X2 = pd.DataFrame(X)
X2.columns = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15"]

fwd_r = forward(X2,Y2)
codnames = cov_order(X2.columns.values)

def test_probability():    
    assert 0. <= np.all(pval_comp(X2.shape[1])) <= 1