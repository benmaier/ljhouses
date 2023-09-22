import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma, kstest as kstestscipy
from scipy.spatial import KDTree
from scipy.special import gammainc, factorial
from tqdm import tqdm
from numpy import sqrt, exp, pi, log
from scipy.special import erf


def ks_dist(x,F):
    x = np.sort(x)
    N = x.shape[0]
    E = np.arange(N) / N
    return np.abs(E - F(x)).max()

def prep_positions(positions, center):
    c = np.array(center,dtype=float).reshape(2)
    x = positions - c[None,:]
    norm = np.linalg.norm(x, axis=1)
    return x, norm

def KSdist_erlang(center, positions, return_R=False):
    x, r = prep_positions(positions, center)
    _r = r.mean()
    R = _r/2
    #KSdist = np.max(np.abs(np.arange(N)/N - get_cdf(r, 2, R)))
    KSdist = ks_dist(r, lambda x: get_cdf(x,2,R))
    if return_R:
        return KSdist, R
    else:
        return KSdist

def KSdist_expon(center, positions, return_R=False):
    x, r = prep_positions(positions, center)
    _r = r.mean()
    R = _r
    #KSdist = np.max(np.abs(np.arange(N)/N - get_cdf(r, 1, R)))
    KSdist = ks_dist(r, lambda x: get_cdf(x,1,R))
    if return_R:
        return KSdist, R
    else:
        return KSdist


def KSdist_truncnorm(center, positions, return_R=False):
    x, r = prep_positions(positions, center)
    mu, sigma = max_likelihood_estimation_trunc(r)
    KSdist = ks_dist(r, lambda x: truncated_normal_cdf(x,mu, sigma))
    if return_R:
        return KSdist, (mu, sigma)
    else:
        return KSdist



def minimize_KS(which_model, positions, showprogress=True, N_subsample=100, add_center_of_mass=True):
    unique_pos = np.unique(positions, axis=0)

    n = unique_pos.shape[0]
    itr = unique_pos

    if N_subsample is not None and N_subsample < n:
        itr = itr[np.random.choice(n,size=min(n,N_subsample),replace=False)]

    if add_center_of_mass:
        CoM = np.mean(positions,axis=0).reshape(1,2)
        #cand = np.repeat(CoM.reshape(1,2),20,axis=0)
        itr = np.concatenate((itr, CoM),axis=0)

    if showprogress:
        itr = tqdm(itr)

    if which_model == 'erlang':
        KSdist = KSdist_erlang
    elif which_model == 'expon':
        KSdist = KSdist_expon
    elif which_model == 'truncnorm':
        KSdist = KSdist_truncnorm

    min_KSd = np.inf
    Ropt = None
    xopt = None
    for pos in itr:
        x0, y0 = pos

        res = minimize(KSdist, x0=(x0,y0), args=(positions,), method='nelder-mead')
        thisxopt = res.x
        KSd, R = KSdist(thisxopt, positions, return_R=True)
        if KSd < min_KSd:
            min_KSd = KSd
            Ropt = R
            xopt = thisxopt


    return min_KSd, Ropt, xopt

def truncated_normal_cdf(x, mu, sigma):
    return (erf((1/2)*sqrt(2)*mu/sigma) - erf((1/2)*sqrt(2)*(mu - x)/sigma))/(erf((1/2)*sqrt(2)*mu/sigma) + 1)

def truncated_normal_pdf(x, mu, sigma):
    return sqrt(2)*exp(-1/2*(mu - x)**2/sigma**2)/(sqrt(pi)*sigma*(erf((1/2)*sqrt(2)*mu/sigma) + 1))

def minus_logLT_over_N(pars, x, x_2):
    mu, sigma = pars
    return -1/2*mu**2/sigma**2 + mu*x/sigma**2 - log(sigma) + log((erf((1/2)*sqrt(2)*mu/sigma) + 1)**(-1.0)) - 1/2*log(pi) + (1/2)*log(2) - 1/2*x_2/sigma**2

def dLTdmu(pars, x, x_2):
    mu, sigma = pars
    return -mu/sigma**2 - sqrt(2)*exp(-1/2*mu**2/sigma**2)/(sqrt(pi)*sigma*(erf((1/2)*sqrt(2)*mu/sigma) + 1)) + x/sigma**2

def dLTdsigma(pars, x, x_2):
    mu, sigma = pars
    return mu**2/sigma**3 + sqrt(2)*mu*exp(-1/2*mu**2/sigma**2)/(sqrt(pi)*sigma**2*(erf((1/2)*sqrt(2)*mu/sigma) + 1)) - 2*mu*x/sigma**3 - 1/sigma + x_2/sigma**3

def gradTrunc(*args):
    return -np.array([dLTdmu(*args),
                      dLTdsigma(*args),
                    ])

def max_likelihood_estimation_trunc(x):
    x2 = (x**2).mean()
    x = x.mean()
    mu0 = x
    sig0 = np.sqrt(x2-x**2)
    minus_logLT_over_N((mu0,sig0),x,x2)
    result = minimize(fun=minus_logLT_over_N,x0=(mu0,sig0),args=(x,x2),jac=gradTrunc,bounds=[(0,np.inf),(0,np.inf)])
    return result.x



def get_cdf(r, k, R):
    return gammainc(k, r/R)/factorial(k-1)

def get_erlang_sample(N, R, phimin=0, phimax=2*np.pi):
    dphi = phimax-phimin
    phi = dphi*np.random.rand(N) + phimin
    c = np.cos(phi)
    s = np.sin(phi)
    r = gamma.rvs(a=2, scale=R, size=N)

    pos = np.zeros((N,2))
    pos[:,0] = r*c
    pos[:,1] = r*s

    return pos

def full_analysis(pos,extent='data',axs=None):

    import matplotlib.pyplot as pl
    no_axes_provided = axs is None

    if no_axes_provided:
        fig, ax = pl.subplots(1,1,figsize=(7,7))
        axs = [ax]
    else:
        ax = axs[0]

    pl.sca(axs[0])
    pl.axis('equal')
    pl.axis('off')

    x, r = prep_positions(pos, [0,0])

    #R = r.mean()/2

    pl.plot(x[:,0],x[:,1],'.',alpha=0.1)
    #pl.show()


    minKSexp, Rexp, Cexp = minimize_KS('expon', x, N_subsample=100)
    minKSerl, Rerl, Cerl = minimize_KS('erlang', x, N_subsample=100)
    minKStrc, Rtrc, Ctrc = minimize_KS('truncnorm', x, N_subsample=100)
    KSerlExp, RerlExp = KSdist_erlang(Cexp, x, return_R=True)

    CoM = x.mean(axis=0)
    print(minKSexp, Rexp, Cexp,'CoM=', CoM)
    print(minKSerl, Rerl, Cerl)
    print(-KSerlExp, RerlExp, Cexp)

    pl.plot(Cexp[:1], Cexp[1:],'X')
    pl.plot(Cerl[:1], Cerl[1:],'P')
    pl.plot(CoM[:1], CoM[1:],'o',label='CoM')
    pl.plot(Ctrc[:1], Ctrc[1:],'d',label='Trunc. normal')

    if extent is not None:
        if extent == 'data':
            extent = [
                    pos[:,0].min(),
                    pos[:,0].max(),
                    pos[:,1].min(),
                    pos[:,1].max()
                ]
        X = np.linspace(extent[0],extent[1],101)
        Y = np.linspace(extent[2],extent[3],101)
        _X, _Y = np.meshgrid(X, Y)

        for KSdist in [KSdist_expon, KSdist_erlang, KSdist_truncnorm]:
            if no_axes_provided:
                fig, ax = pl.subplots(1,1,figsize=(8,8))
                axs.append(ax)
            ax = axs[-1]
            pl.sca(ax)
            pl.axis('equal')
            pl.plot(x[:,0], x[:,1],'.',mec='k',mfc='w')
            pl.plot(Cexp[:1], Cexp[1:],'X',mec='w',ms=8)
            pl.plot(Cerl[:1], Cerl[1:],'P',mec='w',ms=8)
            pl.plot(CoM[:1], CoM[1:],'o',mec='w',ms=8)
            pl.plot(Ctrc[:1], Ctrc[1:],'d',mec='w',ms=8)

            Z = np.zeros((101,101))

            for i, _x in enumerate(X):
                for j, _y in enumerate(Y):
                    mlogL = KSdist((_x,_y), x)
                    Z[j,i] = mlogL

            #pl.imshow(Z[:,::-1],extent=(X[0],X[-1],Y[0],Y[-1]))
            #pl.imshow(Z,extent=(X[0],X[-1],Y[-1],Y[0]))
            pl.contourf(_X, _Y, Z)

            pl.xlim(X[0],X[-1])
            pl.ylim(Y[0],Y[-1])

    if no_axes_provided:
        fig, ax = pl.subplots(2,5,figsize=(12,7))
        axs.append(ax)
    ax = axs[-1]

    rth = np.linspace(0,10*max(Rerl,Rexp),1001)

    for i, cent in enumerate([
            [0,0],
            Cexp,
            Cerl,
            CoM,
            Ctrc,
        ]):

        _x, _r = prep_positions(pos, cent)
        #_r = np.sort(_r)

        KSerl, Rerl = KSdist_erlang(cent, x, return_R=True)
        KSexp, Rexp = KSdist_expon(cent, x, return_R=True)
        KStrc, Rtrc = KSdist_truncnorm(cent, x, return_R=True)

        KSresErl = kstestscipy(_r, lambda x: get_cdf(x,2,Rerl))
        KSresExp = kstestscipy(_r, lambda x: get_cdf(x,1,Rexp))
        KSresTrc = kstestscipy(_r, lambda x: truncated_normal_cdf(x, *Rtrc))

        #N = len(_r)
        #cdf = 1/N * np.arange(0,N)
        #print(np.max(np.abs(cdf-get_cdf(_r,2,Rerl))))
        #print(np.max(np.abs(cdf-get_cdf(_r,1,Rexp))))

        title = ['center: (0,0)','C_exp','C_erl','CoM', 'C_trc'][i]

        print(title, f"{KSresErl=}")
        print(title, f"{KSresExp=}")
        print(title, f"{KSresTrc=}")


        a = ax[0,i]

        a.set_title(title)
        pdf, be, _ = a.hist(_r,bins=101,density=True)
        a.plot(rth, np.exp(-rth/Rexp)/Rexp, label=f'Expon.')
        a.plot(rth, rth*np.exp(-rth/Rerl)/Rerl**2, label=f'Erlang')
        a.plot(rth, truncated_normal_pdf(rth, *Rtrc), label=f'Trunc. normal')

        a = ax[1,i]

        a.set_title(title)
        cdf =  np.cumsum(pdf*np.diff(be))
        print(cdf[0])
        a.step(be[1:],cdf)
        a.plot(rth, get_cdf(rth, 1, Rexp), label=f'KS = {KSexp:4.3f}')
        a.plot(rth, get_cdf(rth, 2, Rerl), label=f'KS = {KSerl:4.3f}')
        a.plot(rth, truncated_normal_cdf(rth, *Rtrc), label=f'KS = {KStrc:4.3f}')

        a.legend()



if __name__ == "__main__":

    import matplotlib.pyplot as pl

    np.random.seed(1)

    N = 2_000
    R = 1
    pos = get_erlang_sample(N, R, phimin=np.pi*1.5)
    full_analysis(pos,extent=[-0.2,0.2,-0.2,0.2])


    pl.show()
