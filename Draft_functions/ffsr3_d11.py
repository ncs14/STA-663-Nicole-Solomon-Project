""" Pseudocode for Fast FSR algorithm """

""" Date: 4/24/15 
    Modified: Improve missing value adjustments (code to test this comes after this cell) """


""" Data type check """
def df_type(dat):
    
    ### Input params:
    #   dat = dataset whose type is to be checked / transformed
    
    ### Output:
    #   error msg or True boolean
    
    import numpy as np
    import pandas as pd
    
    if isinstance(dat,pd.DataFrame)==False and isinstance(dat,np.ndarray)==False:
        raise Exception("Data must be pandas DataFrame")
    else:
        return True


    
""" p-value computation function """
def pval_comp(max_size=None):
    
    ### Input params:
    #   max_size = integer max no. of vars in final model (largest model size desired)
    
    ### Output:
    # array of p-values of each covariate at its given entry step
    
    ### NOTE: fwd should be a global-env R object (requires running 'forward' fcn prior to this fcn) ###
    
    import numpy as np
    import scipy.stats as st
    import rpy2.robjects as ro
    
    # Pull RSS values & num_obs from fwd_proc object
    rss = np.array(ro.r('fwd$rss'))
    N = np.array(ro.r('fwd$nn'))
    
    if max_size==None:
        max_size = len(rss)-1
    
    # vector of model sizes
    sizes = np.arange(max_size)+1
    
    # compute the F stats as defined above where p_f - p_r = 1 for each iteration
    fstats = (rss[0:max_size] - rss[1:(max_size+1)]) / (rss[1:(max_size+1)] / (N - (sizes+1)))
    
    # return the p-values by comparing these stats to the F distn: F(1, n - p_f)
    return 1 - st.f.cdf(fstats, 1, N-(sizes+1))



""" Covariate model entry order """
def cov_order(xcolnames,max_size=None,col_incl=None):
    
    # Input params:
    #   xcolnames = array of names of covariates (same order as columns in original dataset)
    #   max_size  = integer max no. of vars in final model (largest model size desired)
    #   col_incl  = array vector of columns to forcefully include in all models
    
    ### Output:
    # array of covariate names sorted according to order of entry into the model
    
    ### NOTE: fwd should be a global-env R object (requires running 'forward' fcn prior to this fcn) ###
    
    import numpy as np
    import rpy2.robjects as ro
    
    if max_size==None:
        max_size = len(xcolnames)
        
    ### Pull the cov entry order
    vorder = ro.r('fwd$vorder[-1]') # remove intercept
    vorder = vorder[0:max_size] # keep only the max model size number of covs
    
    ### Shift these values down by two (one to exclude intercept, one to make python indices)
    vorderinds = np.array(vorder)-2
    
    ### Rearrange the var order st forced vars are at start of list
    if col_incl==None:
        col_incl = np.arange(max_size)+1
    keep = xcolnames[[col_incl-1]] # pull var names of those vars forced into model (this is an array)
    poss = [x for x in xcolnames if x not in keep] # pull var names of those not forced in (this is a list)
    col_names = np.array(list(keep)+poss) # = rearranged array of varnames w/forced-in vars at start of list
    
    ### Sort the columns of X in order to obtain the var names in the entry order
    return col_names[vorderinds[::]]

    

""" Forward selection function """
def forward(x,y,max_size=None,col_incl=None):
    
    ### Input params:
    #   x        = python dataframe of original p covariates, n x p
    #   y        = python outcome dataframe, n x 1
    #   max_size = integer max no. of vars in final model (largest model size desired)
    #   col_incl = array vector of columns to forcefully include in all models
    
    ### Output:
    # regsubsets R object -- the raw full output of the forward selection proc
    
    ### Load python packages to call R functions
    import rpy2.robjects as ro
    import pandas.rpy.common as com
    
    ### Convert x and y to R matrices <-- MAKE SURE x,y input == DATAFRAMES (or else change them to df's)!!!
    ### and declare as R objects in global environment
    ro.globalenv['x2'] = com.convert_to_r_matrix(x)
    ro.globalenv['y2'] = com.convert_to_r_matrix(y)
    if max_size==None:
        max_size = x.shape[1]
    ro.globalenv['maxv'] = ro.Vector(max_size)
    if col_incl==None:
        ro.r('coli=NULL')
    else:
        ro.globalenv['coli'] = ro.FloatVector(col_incl[:])
    
    ### Perform forward selection with regsubsets function
    ro.globalenv['fwd'] = ro.r('leaps::regsubsets(x=x2,y=y2,method="forward",nvmax=maxv,force.in=coli)')
    
    
    
""" Gamma computation """
def gamma_F(pvs, ncov, max_size=None):
    
    ### Input params:
    #   pvs      = vector of p-values (monotonically increasing) from forward sel procedure
    #   ncov     = integer total number of covariates in data
    #   max_size = integer max no. of vars in final model (largest model size desired)
    
    ### Output:
    # array of gamma_F values
    
    import numpy as np
    
    if max_size==None:
        max_size = ncov
        
    # Create indices == model size at given step, call this S
    S = np.arange(max_size)+1
    
    # gamma_F_i = p_s_i * (ncov - S_i) / (1 + S_i)
    g_F = pvs * (ncov - S) / (1 + S)
    
    # Check for duplicate p-values
    dups = list(set([x for x in list(pvs) if list(pvs).count(x) > 1]))
    for i in range(len(dups)): g_F[pvs==dups[i]] = min(g_F[pvs==dups[i]])
    
    # if table run on all vars, the last gamma = 0,
    #  instead set equal to the last pv_mono == final rate of unimp var inclusion
    if(g_F[-1]==0): 
        g_F[-1]=pvs[-1]
    
    return g_F

    
    
""" Alpha computation for model selection """
def alpha_F(g0, ncov, max_size=None):
    
    ### Input params:
    #   g0       = float pre-specified FSR (gamma0)
    #   ncov     = integer total number of covariates in data
    #   max_size = integer max no. of vars in final model (largest model size desired)
    
    ### Output:
    # array of alpha_F values
    
    import numpy as np
    
    if max_size==None:
        max_size = ncov
        
    # Create indices == model size at given step, call this S
    S = np.arange(max_size)+1
    
    # alpha_F_i = gamma_0 * (1 + S_i) / (ncov - S_i)
    alpha_F = g0 * (1 + S) / (ncov - S)
    
    # if table run on all vars, the last alpha = inf
    #  instead set equal to 1 == include all vars
    alpha_F[np.isinf(alpha_F)] = 1.
    
    return alpha_F        
    
    
    
""" Alpha computation for specific gamma """
def alpha_F_g(g, gf, ncov):
    
    ### Input params:
    #   g    = float or vector (length k) of specified FSR at which to compute alpha
    #   gf   = vector gamma_F's computed from gamma0, pv_sorted
    #          used to compute largest size model (S) for which gamma_F < g
    #   ncov = integer of total number covariates in data
    
    ### Output:
    # integer alpha_F value
    
    import numpy as np
    
    ### Compute model size for gf closest to (but still <) g
    #S = np.array([max(np.which(x<=y)) for x in gf y in g])+1
    if isinstance(g,np.ndarray): # if g is a vector
        s_s = [np.where(gf>y) for y in g]
        S = np.array([min(x[0]) for x in s_s])
        return g * (1 + S) / (ncov - S)
    else: # if g is a number
        S = min(np.where(gf>g)[0])
        return g * (1 + S) / (ncov - S)


    
""" Beta-hat computation for specific gamma """
def beta_est(x, y, g, gf, vname):
    
    ### Input params:
    #   x      = python dataframe of original p covariates, n x p
    #   y      = python outcome dataframe, n x 1
    #   g      = float of specified FSR at which to compute alpha
    #   gf     = vector gamma_F's computed from gamma0, pv_mono
    #            used to compute largest size model (S) for which gamma_F < g
    #   vname  = ordered vector of names of vars entered into model under forward selection
    
    ### Output:
    # array of estimated parameters
    
    import numpy as np
    import statsmodels.api as sm
    
    ### Compute model size corresponding to g
    S = min(np.where(gf>g)[0])

    ### Pull the cov names of those vars included in the above size model
    modvars = vname[:S]

    ### Fit the linear model using the selected model vars
    fit = sm.OLS(y,x.loc[:,list(modvars)]).fit()
    betaout = pd.DataFrame([fit.params,fit.bse]).T
    betaout.columns = ['beta','beta_se']
    
    return betaout

    
    
""" FSR Results Table """
def fsrtable(size, vname, p_orig, p_mono, alphaf, gammaf, prec_f='.4f'):
    
    ### Input params:
    #   size   = model size at each step of forward sel proc                   [S]
    #   vname  = variable name that entered at each step (num vars = p)        [Var]
    #   p_orig = p-values at each step                                         [p]
    #   p_mono = ascending p-values                                            [p_s]
    #   alphaf = alpha-to-enter (p-value cutoff) for model entry at each step  [alpha_F]
    #   gammaf = FSR at each step                                              [gamma_F]
    #   prec_f = string of precision (num digits) desired in FSR output table
    
    ### Output:
    # table of [S   Var   p   p_s   alpha_F   gamma_F], dim = num_steps(== p) x 6
    
    import numpy as np
    import pandas as pd
    
    ### Round all arrays
    p_od = [format(x,prec_f) for x in p_orig]
    p_md = [format(x,prec_f) for x in p_mono]
    ad = [format(x,prec_f) for x in alphaf]
    gd = [format(x,prec_f) for x in gammaf]
    
    ### Combine the arrays
    tab = pd.DataFrame([size,vname,p_od,p_md,ad,gd]).T
    tab.columns = ['S','Var','p','p_m','alpha_F','gamma_F']
    
    return tab
    
    
    
""" FastFSR function """
def ffsr(dat,g0=0.05,betaout=False,gs=None,max_size=None,var_incl=None,bag=False,prec_f='.4f'):
    
    ### Input params:
    #   dat      = python dataframe of original p covariates, 1 outcome (in first col.): n x p+1
    #   g0       = float pre-specified FSR of interest ("gamma0")
    #   betaout  = boolean of whether to include estimated betahats from final selected model
    #   gs       = float or vector of gamma's at which to specifically compute alpha_F
    #   max_size = integer of largest model size == max num vars to incl in final model (default = num covs in dataset)
    #   var_incl = array of cols corresponding to those vars to force into model
    #   bag      = boolean of whether to output FSR table (non-bagging results) or reduced output for bagging purposes
    #   prec_f   = string of precision (num digits) desired in FSR output table (string to be given to 'format' python fcn)
    
    ### Output: 
    #      (note: gamma = FSR, gamma_0 = pre-specified/desired FSR)
    # Table of [S   Var   p   p_s   alpha_F   gamma_F], dim = num_steps(== p) x 6
    #   S:       model size at given step
    #   Var:     name of var that entered at given step
    #   p:       p-value of var that entered at given step
    #   p_m:     mono. inc. p-value (vector or original p-values arranged to be monotonically increasing)
    #   alpha_F: cutoff value for model entry given gamma_0 and current p_s value
    #   gamma_F: FSR given current alpha_F and model size (== step num)
    #       and
    #   vector of alpha_F's for specified gamma's (g)
    #       and
    #   vector of estimated beta param's for final model (based on g0)

    import numpy as np
    import pandas as pd
    
    ### Clean and check data - make sure dat = pandas dataframes or else convert them
    if bag==False:
        if df_type(dat)==True:
            if isinstance(dat,pd.DataFrame):
                d = dat.copy()
            else:
                if isinstance(dat,np.ndarray):
                    d = pd.DataFrame(dat)
                    vnum = list(np.arange(d.shape[1])+1)
                    vchr = list(np.repeat("V",d.shape[1]))
                    d.columns = [a + str(b) for a,b in zip(vchr,vnum)]
        else:
            return df_type(dat)
        
        ### Remove missing values
        d.dropna(inplace=True)
        
        ### Check that p < n to ensure regression solutions
        if (d.shape[1]-1) >= d.shape[0]:
            raise Exception("N must be > p for valid regression solutions")
    else:
        d = dat.copy()
    
    ### If max model size not specified, select all possible cov.s
    if max_size==None:
        max_size = d.shape[1]-1
        
    y, x = pd.DataFrame(d.iloc[:,0]), d.iloc[:,1:]
        
    ### Perform forward selection
    fwd_sel = forward(x, y, max_size, var_incl)
    
    ### Save order of covariate entry into model
    cov_entry_order = cov_order(d.columns.values[1:], max_size, var_incl)
    
    ### Compute p-value of each covariate entering the model
    p_orig = pval_comp(max_size)
    
    ### Arrange p-values in mono. inc. order
    p_mono = np.array([max(p_orig[:(i+1)]) for i in range(len(p_orig))])
        
    ### Gamma_F computation
    g_F = gamma_F(p_mono, d.shape[1], max_size)
    
    ### Check if betaout desired, if so compute beta_hat of model corresponding to specific gamma0
    if betaout==True or bag==True:
        betahats = beta_est(x, y, g0, g_F, cov_entry_order)
        
    ### Check if bagging desired
    if bag==False: 
        ### Alpha_F computation for all steps in fwd sel proc
        a_F = alpha_F(g0, d.shape[1]-1, max_size)
        
        ### Model size
        S = np.arange(max_size)+1
        
        ### Combine S, Cov_names, p-vals, sorted p-vals, alpha_F, gamma_F into table
        fsr_results = fsrtable(S, cov_entry_order, p_orig, p_mono, a_F, g_F)
        
        ### Return selected output: FSR table (+ betahat) (+ alpha_specific)
        if gs!=None: 
            ### Compute alpha_F for specific gammas (gs)
            if betaout==True:
                return fsr_results, betahats, alpha_F_g(gs, g_F, d.shape[1]-1)
            else:
                return fsr_results, alpha_F_g(gs, g_F, d.shape[1]-1)
        else:
            if betaout==True:
                return fsr_results, betahats
            else:
                return fsr_results
    else:
        return betahats, alpha_F_g(g0, g_F, d.shape[1]-1), len(betahats)

    
    
def bagfsr(dat,g0,B=200,max_s=None,v_incl=None,prec=4):
    
    ### Input params:
    #   dat    = python dataframe of original p covariates, 1 outcome (in first col.): n x p+1
    #   g0     = float pre-specified FSR of interest ("gamma0")
    #   B      = integer of number of bagged samples
    #   max_s  = integer of largest model size == max num vars to incl in final model (default = num covs in dataset)
    #   v_incl = array of cols corresponding to those vars to force into model
    #   prec   = integer of precision (num digits) desired in beta-hat parameter estimates of final model
    
    ### Output: 
    #   Mean of betahats
    #   SEs of betahats
    #   Avg alpha-to-enter
    #   Avg model size
    #   Prop of times each var included in model
    
    import numpy as np
    import pandas as pd
    
    ### Clean and check data - make sure X, Y = pandas dataframes or else convert them
    if df_type(dat)==True:
        if isinstance(dat,pd.DataFrame):
            d = dat.copy()
        else:
            if isinstance(dat,np.ndarray):
                d = pd.DataFrame(dat)
                vnum = list(np.arange(d.shape[1])+1)
                vchr = list(np.repeat("V",d.shape[1]))
                d.columns = [a + str(b) for a,b in zip(vchr,vnum)]
    else:
        return df_type(dat)
    
    ### Remove missing values
    d.dropna(inplace=True)
    
    ### check that p < n to ensure regression solutions
    if (d.shape[1]-1) >= d.shape[0]:
        raise Exception("N must be > p for valid regression solutions")
    
    ### Create array to keep track of number of times vars enter model
    nentries = pd.DataFrame(np.zeros(d.shape[1]-1),index=d.columns.values[1:])
    
    ### Create array to store all estimated coefficients, ses, alphas, sizes
    allbetas = pd.DataFrame(np.zeros([B,(d.shape[1]-1)]),columns=d.columns.values[1:])
    allses = allbetas.copy()
    alphas = []
    sizes = []
    np.random.seed(1234)
    
    ### Bagging loops
    for i in range(B):

        # Draw with replacement from rows of data
        n_row = d.shape[0]
        rand_row = np.random.randint(0,n_row,n_row)
        newdat = d.iloc[rand_row,:]
        newdat.index = np.arange(n_row)+1
        
        ### Obtain FSR results
        fsrout = ffsr(newdat.iloc[:,1:],pd.DataFrame(newdat.iloc[:,0]),g0,bag=True,max_size=max_s,var_incl=v_incl)
        allbetas.loc[i,fsrout[0].index.values] = fsrout[0].iloc[:,0]
        allses.loc[i,fsrout[0].index.values] = fsrout[0].iloc[:,1]
        alphas.append(fsrout[1])
        sizes.append(fsrout[2])

        ### Update counts num times var included
        nentries.loc[fsrout[0].index[np.abs(np.around(fsrout[0].iloc[:,0],prec))>0]] += 1
        
    ### Compute averages
    avgbeta = np.around(allbetas.mean(axis=0),prec) # mean across rows / colmeans == mean of each cov's betahat
    avgse = np.around(allses.mean(axis=0),prec)
    avgalpha = np.mean(alphas)
    avgsize = np.mean(sizes)
    var_props = nentries/float(B)
    cov_res = pd.concat([avgbeta,avgse,var_props],axis=1)
    cov_res.columns = ['betahat','betase','prop_incl']
    
    return cov_res, avgalpha, avgsize
    
    

# Notes: 
# 1. appropriate transformations are expected to have been applied prior to utilization of FSR algorithm