
from ffsr import ffsr
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

### Read in NCAA2 data from NCSU webpage
ncaadata = pd.read_csv("ncaa_data2.txt", delim_whitespace=True, skipinitialspace=True)

# move outcome variable to first column
cols = ncaadata.columns.tolist()
cols = cols[-1:] + cols[:-1]

ncaa2 = pd.DataFrame(ncaadata[cols],dtype='float')

print "Python Results:\n"

print ffsr(ncaa2,0.05,var_incl=np.array([12,3,5])).fsres