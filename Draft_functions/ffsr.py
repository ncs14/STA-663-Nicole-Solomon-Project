""" Pseudocode for Fast FSR algorithm """

""" Date: 4/8/15 
    Modified: Profile primary function ffsr """


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
def pval_comp(max_size=None,prec_f=4):
    
    ### Input params:
    #   max_size = integer max no. of vars in final model (largest model size desired)
    #   prec_f   = integer of precision (num digits) desired in FSR output table
    
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
    fstats = (rss[0:max_size] - rss[1:(max_size+1)]) / (rss[1:(max_size+1)] / (N - sizes))
    
    # return the p-values by comparing these stats to the F distn: F(1, n - p_f)
    return np.around(1 - st.f.cdf(fstats, 1, N-sizes),prec_f)



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
def gamma_F(pvs, ncov, max_size=None, prec_f=4):
    
    ### Input params:
    #   pvs      = vector of p-values (sorted or unsorted) from forward sel procedure
    #   ncov     = integer total number of covariates in data
    #   max_size = integer max no. of vars in final model (largest model size desired)
    #   prec_f   = integer of precision (num digits) desired in FSR output table
    
    ### Output:
    # array of gamma_F values
    
    import numpy as np
    
    # sort pvalues to be monotonically increasing 
    pv_s = np.sort(pvs)
    
    if max_size==None:
        max_size = ncov
        
    # Create indices == model size at given step, call this S
    S = np.arange(max_size)+1
    
    # gamma_F_i = p_s_i * (ncov - S_i) / (1 + S_i)
    g_F = pv_s * (ncov - S) / (1 + S)
    
    # Check for duplicate p-values
    dupps = list(set([x for x in list(pv_s) if list(pv_s).count(x) > 1]))
    g_F[pv_s==dupps] = min(g_F[pv_s==dupps])
    
    # if table run on all vars, the last gamma = 0,
    #  instead set equal to the last pv_sort == final rate of unimp var inclusion
    if(g_F[-1]==0): 
        g_F[-1]=pv_s[-1]
    
    return np.around(g_F,prec_f)

    
    
""" Alpha computation for model selection """
def alpha_F(g0, ncov, max_size=None, prec_f=4):
    
    ### Input params:
    #   g0       = float pre-specified FSR (gamma0)
    #   ncov     = integer total number of covariates in data
    #   max_size = integer max no. of vars in final model (largest model size desired)
    #   prec_f   = integer of precision (num digits) desired in FSR output table
    
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
    
    return np.around(alpha_F,prec_f)        
    
    
    
""" Alpha computation for specific gamma """
def alpha_F_g(g, gf, ncov, prec_f=4):
    
    ### Input params:
    #   g    = float or vector (length k) of specified FSR at which to compute alpha
    #   gf   = vector gamma_F's computed from gamma0, pv_sorted
    #          used to compute largest size model (S) for which gamma_F < g
    #   ncov = integer of total number covariates in data
    #   prec_f   = integer of precision (num digits) desired in FSR output table
    
    ### Output:
    # integer alpha_F value
    
    import numpy as np
    
    ### Compute model size for gf closest to (but still <) g
    #S = np.array([max(np.which(x<=y)) for x in gf y in g])+1
    if isinstance(g,np.ndarray): # if g is a vector
        s_s = [np.where(gf>y) for y in g]
        S = np.array([min(x[0]) for x in s_s])
        return np.around(g * (1 + S) / (ncov - S),prec_f)
    else: # if g is a number
        S = min(np.where(gf>g)[0])
        return round(g * (1 + S) / (ncov - S),prec_f)


    
""" Beta-hat computation for specific gamma """
def beta_est(x, y, g, gf, vname):
    
    ### Input params:
    #   x      = python dataframe of original p covariates, n x p
    #   y      = python outcome dataframe, n x 1
    #   g      = float of specified FSR at which to compute alpha
    #   gf     = vector gamma_F's computed from gamma0, pv_sorted
    #            used to compute largest size model (S) for which gamma_F < g
    #   vname  = ordered vector of names of vars entered into model under forward selection
    #   bag    = boolean for whether output is used with bagging - output all betaest's
    
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
def fsrtable(size, vname, p_orig, p_sort, alphaf, gammaf):
    
    ### Input params:
    #   size   = model size at each step of forward sel proc                   [S]
    #   vname  = variable name that entered at each step (num vars = p)        [Var]
    #   p_orig = p-values at each step                                         [p]
    #   p_sort = ascending p-values                                            [p_s]
    #   alphaf = alpha-to-enter (p-value cutoff) for model entry at each step  [alpha_F]
    #   gammaf = FSR at each step                                              [gamma_F]
    
    ### Output:
    # table of [S   Var   p   p_s   alpha_F   gamma_F], dim = num_steps(== p) x 6
    
    import pandas as pd
    
    ### Convert all arrays to dataframes
    sized = pd.DataFrame(size)
    vnamed = pd.DataFrame(vname)
    p_od = pd.DataFrame(p_orig)
    p_sd = pd.DataFrame(p_sort)
    ad = pd.DataFrame(alphaf)
    gd = pd.DataFrame(gammaf)
    
    ### Combine the arrays
    tab = pd.concat([sized,vnamed,p_od,p_sd,ad,gd],axis=1)
    tab.columns = ['S','Var','p','p_s','alpha_F','gamma_F']
    
    return tab
    
    
    
""" FastFSR function """
def ffsr(X,Y,g0=0.05,betaout=False,gs=None,max_size=None,var_incl=None,bag=False,prec_f=4,prec_b=6):
    
    ### Input params:
    #   x        = python dataframe of original p covariates, n x p
    #   y        = python outcome dataframe, n x 1
    #   g0       = float pre-specified FSR of interest ("gamma0")
    #   betaout  = boolean of whether to include estimated betahats from final selected model
    #   gs       = float or vector of gamma's at which to specifically compute alpha_F
    #   max_size = integer of largest model size == max num vars to incl in final model (default = num covs in dataset)
    #   var_incl = array of cols corresponding to those vars to force into model
    #   bag      = boolean of whether to output FSR table (non-bagging results) or reduced output for bagging purposes
    #   prec_f   = integer of precision (num digits) desired in FSR output table
    #   prec_b   = integer of precision (num digits) desired in beta-hat parameter estimates of final model
    
    ### Output: 
    #      (note: gamma = FSR, gamma_0 = pre-specified/desired FSR)
    # Table of [S   Var   p   p_s   alpha_F   gamma_F], dim = num_steps(== p) x 6
    #   S:       model size at given step
    #   Var:     name of var that entered at given step
    #   p:       p-value of var that entered at given step
    #   p_s:     sorted p-value (vector or original p-values sorted in increasing order)
    #   alpha_F: cutoff value for model entry given gamma_0 and current p_s value
    #   gamma_F: FSR given current alpha_F and model size (== step num)
    #       and
    # Vector of alpha_F's for specified gamma's (g)

    import numpy as np
    import pandas as pd
    
    ### Clean and check data - make sure X, Y = pandas dataframes or else convert them
    if df_type(X)==True:
        if isinstance(X,pd.DataFrame):
            x = X.copy()
        else:
            if isinstance(X,np.ndarray):
                x = pd.DataFrame(X)
                vnum = list(np.arange(x.shape[1])+1)
                vchr = list(np.repeat("V",x.shape[1]))
                x.columns = [a + str(b) for a,b in zip(vchr,vnum)]
    else:
        return df_type(X)
    if df_type(Y)==True:
        if isinstance(Y,pd.DataFrame):
            y = Y.copy()
        else:
            if isinstance(Y,np.ndarray):
                y = pd.DataFrame(Y)
    else:
        return df_type(Y)
    
    # remove missing values
    yna = np.isnan(y).any(axis=1)
    xna = np.isnan(x).any(axis=1).reshape(x.shape[0],1)
    anyna = np.array([int(max(a,b)) for a,b in zip(xna,yna)])
    missrow = np.where(anyna==1)[0]
    y = y.drop(y.index[missrow])
    x = x.drop(x.index[missrow])
    
    # check that p < n to ensure regression solutions
    if x.shape[1] >= x.shape[0]:
        raise Exception("N must be > p for valid regression solutions")
    
    ### If max model size not specified, select all possible cov.s
    if max_size==None:
        max_size = x.shape[1]
        
    ### Perform forward selection
    fwd_sel = forward(x, y, max_size, var_incl)
    
    ### Save order of covariate entry into model
    cov_entry_order = cov_order(x.columns.values, max_size, var_incl)
    
    ### Compute p-value of each covariate entering the model
    p_orig = pval_comp(max_size, prec_f)
    
    ### Sort p-values in ascending order
    p_sort = np.sort(p_orig)
        
    ### Gamma_F computation
    g_F = gamma_F(p_sort, x.shape[1], max_size, prec_f)
    
    ### Check if betaout desired, if so compute beta_hat of model corresponding to specific gamma0
    if betaout==True or bag==True:
        betahats = beta_est(x, y, g0, g_F, cov_entry_order)
        
    ### Check if bagging desired
    if bag==False: 
        ### Alpha_F computation for all steps in fwd sel proc
        a_F = alpha_F(g0, x.shape[1], max_size, prec_f)
        
        ### Model size
        S = np.arange(max_size)+1
        
        ### Combine S, Cov_names, p-vals, sorted p-vals, alpha_F, gamma_F into table
        fsr_results = fsrtable(S, cov_entry_order, p_orig, p_sort, a_F, g_F)
        
        ### Return selected output: FSR table (+ betahat) (+ alpha_specific)
        if gs!=None: 
            ### Compute alpha_F for specific gammas (gs)
            if betaout==True:
                return fsr_results, np.around(betahats, prec_b), alpha_F_g(gs, g_F, x.shape[1], prec_f)
            else:
                return fsr_results, alpha_F_g(gs, g_F, x.shape[1], prec_f)
        else:
            if betaout==True:
                return fsr_results, np.around(betahats, prec_b)
            else:
                return fsr_results
    else:
        return betahats, alpha_F_g(g0, g_F, x.shape[1], prec_f), len(betahats)

    
    
def bagfsr(X,Y,g0,B=200,max_s=None,v_incl=None,prec=4):
    
    ### Input params:
    #   X      = python dataframe of original p covariates, n x p
    #   Y      = python outcome dataframe, n x 1
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
    if df_type(X)==True:
        if isinstance(X,pd.DataFrame):
            x = X.copy()
        else:
            if isinstance(X,np.ndarray):
                x = pd.DataFrame(X)
                vnum = list(np.arange(x.shape[1])+1)
                vchr = list(np.repeat("V",x.shape[1]))
                x.columns = [a + str(b) for a,b in zip(vchr,vnum)]
    else:
        return df_type(X)
    if df_type(Y)==True:
        if isinstance(Y,pd.DataFrame):
            y = Y.copy()
        else:
            if isinstance(Y,np.ndarray):
                y = pd.DataFrame(Y)
    else:
        return df_type(Y)
    
    ### Combine data into single dataframe
    dat = pd.concat([y,x],axis=1)
    
    ### Create array to keep track of number of times vars enter model
    nentries = pd.DataFrame(np.zeros(x.shape[1]),index=x.columns.values)
    
    ### Create array to store all estimated coefficients, ses, alphas, sizes
    allbetas = pd.DataFrame(np.zeros([B,x.shape[1]]),columns=x.columns.values)
    allses = allbetas.copy()
    alphas = []
    sizes = []
    
    ### Bagging loops
    for i in range(B):

        # Draw with replacement from rows of data
        np.random.seed(1234)
        n_row = dat.shape[0]
        rand_row = np.random.randint(0,n_row,n_row)
        newdat = dat.iloc[rand_row,:]
        newdat.index = np.arange(n_row)+1
        
        ### Obtain FSR results
        precall = 8
        fsrout = ffsr(newdat.iloc[:,1:],pd.DataFrame(newdat.iloc[:,0]),g0,bag=True,max_size=max_s,var_incl=v_incl)
        allbetas.loc[i,fsrout[0].index.values] = np.around(fsrout[0].iloc[:,0],precall)
        allses.loc[i,fsrout[0].index.values] = np.around(fsrout[0].iloc[:,1],precall)
        alphas.append(fsrout[1])
        sizes.append(fsrout[2])

        ### Update counts num times var included
        nentries.loc[fsrout[0].index[np.abs(np.around(fsrout[0].iloc[:,0],precall))>0]] += 1
        
    ### Compute averages
    avgbeta = allbetas.mean(axis=0) # mean across rows / colmeans == mean of each cov's betahat
    avgse = allses.mean(axis=0)
    avgalpha = np.mean(alphas)
    avgsize = np.mean(sizes)
    var_props = nentries/float(B)
    cov_res = pd.concat([avgbeta,avgse,var_props],axis=1)
    cov_res.columns = ['betahat','betase','prop_incl']
    
    return cov_res, avgalpha, avgsize
    
    

# Notes: 
# 1. appropriate transformations are expected to have been applied prior to utilization of FSR algorithm

# To-do:
# 1. adjust betaest fcn and ffsr to allow for specification of intercept and whether data should be normalized in estimation