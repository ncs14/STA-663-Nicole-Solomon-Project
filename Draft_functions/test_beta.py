
from ffsr import beta_est
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

def test_nox():
    assert_raises(AttributeError,beta_est,None,Y2,0.05,g,X2.columns.values)

def test_noy():
    assert_raises(TypeError,beta_est,X2,None,0.05,g,X2.columns.values)
    
def test_nog0():
    assert_raises(ValueError,beta_est,X2,Y2,None,g,X2.columns.values)
    
def test_nog():
    assert_raises(ValueError,beta_est,X2,Y2,0.05,None,X2.columns.values)

def test_nocovnames():
    assert_raises(TypeError,beta_est,X2,Y2,0.05,g,None)