
from ffsr import bagfsr
import pandas as pd
import numpy as np

### Read in NCAA2 data from NCSU webpage
ncaadata = pd.read_csv("ncaa_data2.txt", delim_whitespace=True, skipinitialspace=True)

# move outcome variable to first column
cols = ncaadata.columns.tolist()
cols = cols[-1:] + cols[:-1]

ncaa2 = pd.DataFrame(ncaadata[cols],dtype='float')

f = bagfsr(ncaa2,0.05)

print "Python Results:\n"
print
print "Mean of estimated alpha-to-enter:", round(f.alpha,4)
print
print "Mean size of selected model:", f.size