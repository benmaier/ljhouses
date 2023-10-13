#change the following according to your needs
import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import digamma, gamma, polygamma
from scipy.optimize import minimize, newton

def pdf(x, R, m):
    return x/R**2 *np.exp(-(x/R)**m)*m / gamma(2/m)

def get_w(m,r):
    rm = (r**m).mean()
    return (2/m/rm)**(1/m)

def get_R(m,r):
    return 1/get_w(m,r)

def dlogLdm(m,r):
    w = get_w(m, r)
    return 1/m + 2/m**2 * digamma(2/m) - np.mean(np.log(w*r)*(w*r)**m)

def d2logLdm2(m,r):
    w = get_w(m, r)
    return -1/m**2 - 4/m**3 * digamma(2/m) - (2/m**2)**2 * polygamma(1,2/m) - np.mean(np.log(w*r)**2 * (w*r)**m)

def fit_pdf(r,m0=0.1):
    """
    Finds R, m of the generalized pairwise-distance distribution
    that best fits the data in ``r`` (maximizes likelihood).
    """
    m = newton(dlogLdm, m0,args=(r,))
    #res = minimize(minus_logL, (m0,), args=(r,), bounds=[(1e-9,np.inf)],)
    #m = res.x[0]
    R = get_R(m, r)
    return R, m


