
import numpy as np
import pandas as pd
from numpy.testing import assert_raises
from ffsr import df_type

def test_df_warn():
    x = list(np.arange(5))
    assert_raises(ValueError,df_type,x)
    
def test_df_df():
    x1 = np.arange(15).reshape(3,5)
    x2 = pd.DataFrame(x1)
    assert df_type(x1)*1+df_type(x2)*1 == 2