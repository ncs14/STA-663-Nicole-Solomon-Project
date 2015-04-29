
import ffsr as f
from numpy.testing import assert_raises

import rpy2.robjects as ro
import pandas.rpy.common as com
from rpy2.robjects.packages import importr

import numpy as np
import pandas as pd

leaps = importr('leaps')
stats = importr('stats')
base = importr('base')

regsub = ro.r('leaps::regsubsets')

np.random.seed(1234)

X = np.random.multivariate_normal(np.zeros(15),np.eye(15),(100))
beta = np.array([0,0,5,6,0,0,4,0,0,0,5,0,0,0,0]).reshape(15,1) # signif betas: 3,4,7,11
Y = X.dot(beta)
Y2 = pd.DataFrame(Y)
X2 = pd.DataFrame(X)
X2.columns = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15"]

fwd_r = f.forward(X2,Y2)

def test_nonames():    
    assert_raises(TypeError,f.cov_order,None)