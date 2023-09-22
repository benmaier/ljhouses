import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma, kstest
from scipy.spatial import KDTree
from scipy.special import gammainc, factorial
from tqdm import tqdm

def prep_positions(positions, center):
    c = np.array(center,dtype=float).reshape(2)
    x = positions - c[None,:]
    norm = np.linalg.norm(x, axis=1)
    return x, norm

def minus_logL_erlang(center, positions, eps=0, return_R=False):
    x, r = prep_positions(positions, center)
    _r = r.mean()
    R = _r/2
    logr = np.log(r+eps).mean()
    logL = logr - 2*np.log(_r/2) - 2
    if return_R:
        return -logL, R
    else:
        return -logL

def minus_logL_expon(center, positions, eps=0, return_R=False):
    x, r = prep_positions(positions, center)
    _r = r.mean()
    R = _r
    logL = -1 - np.log(_r + eps)
    if return_R:
        return -logL, R
    else:
        return -logL


def maximize_logL(which_model, positions, eps=None, showprogress=True, N_subsample=1000, add_center_of_mass=True):
    unique_pos = np.unique(positions, axis=0)

    # if no epsilon is provided, take half the minimum distance between
    # any two points 
    if eps is None:
        T = KDTree(unique_pos)
        r, _ = T.query(unique_pos,k=2)
        r = r[:,1] # ignore first entry, which will be the nodes themselves
        eps = r.min()/2

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
        minus_logL = minus_logL_erlang
    elif which_model == 'expon':
        minus_logL = minus_logL_expon

    maxL = -np.inf
    Ropt = None
    xopt = None
    for pos in itr:
        x0, y0 = pos + np.random.rand(2) * eps
        #x0, y0 = pos

        res = minimize(minus_logL, x0=(x0,y0), args=(positions,0))
        thisxopt = res.x
        m_logL, R = minus_logL(thisxopt, positions, return_R=True)
        if -m_logL > maxL:
            maxL = -m_logL
            Ropt = R
            xopt = thisxopt


    return maxL, Ropt, xopt


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


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    fig, ax = pl.subplots(1,1,figsize=(7,7))
    pl.axis('equal')
    pl.axis('off')

    N = 2_000
    R = 1
    pos = get_erlang_sample(N, R, phimin=np.pi)

    x, r = prep_positions(pos, [0,0])

    print(R, r.mean()/2)

    pl.plot(x[:,0],x[:,1],'.',alpha=0.1)
    #pl.show()


    maxLexp, Rexp, Cexp = maximize_logL('expon', x, eps=None, N_subsample=400)
    maxLerl, Rerl, Cerl = maximize_logL('erlang', x, eps=None, N_subsample=400)
    LerlExp, RerlExp = minus_logL_erlang(Cexp, x, return_R=True)

    CoM = x.mean(axis=0)
    print(maxLexp, Rexp, Cexp,'CoM=', CoM)
    print(maxLerl, Rerl, Cerl)
    print(-LerlExp, RerlExp, Cexp)

    pl.plot(Cexp[:1], Cexp[1:],'X')
    pl.plot(Cerl[:1], Cerl[1:],'P')
    pl.plot(CoM[:1], CoM[1:],'o')

    X = np.linspace(-0.2,0.2,101)
    Y = X
    _X, _Y = np.meshgrid(X, Y)

    for minus_logL in [minus_logL_expon, minus_logL_erlang]:
        fig, ax = pl.subplots(1,1,figsize=(8,8))
        pl.axis('equal')
        pl.plot(x[:,0],x[:,1],'.',mec='k')
        pl.plot(Cexp[:1], Cexp[1:],'X',mec='k')
        pl.plot(Cerl[:1], Cerl[1:],'P',mec='k')
        pl.plot(CoM[:1], CoM[1:],'o',mec='k')

        Z = np.zeros((101,101))

        for i, _x in enumerate(X):
            for j, _y in enumerate(Y):
                mlogL = minus_logL((_x,_y), x)
                Z[j,i] = mlogL

        #pl.imshow(Z[:,::-1],extent=(X[0],X[-1],Y[0],Y[-1]))
        #pl.imshow(Z,extent=(X[0],X[-1],Y[-1],Y[0]))
        pl.contourf(_X, _Y, Z)

        pl.xlim(X[0],X[-1])
        pl.ylim(Y[0],Y[-1])

    fig, ax = pl.subplots(2,4,figsize=(12,7))

    rth = np.linspace(0,10*R,1001)

    for i, cent in enumerate([
            [0,0],
            Cexp,
            Cerl,
            CoM,
        ]):

        _x, _r = prep_positions(pos, cent)
        _r = np.sort(_r)

        Lerl, Rerl = minus_logL_erlang(cent, x, return_R=True)
        Lexp, Rexp = minus_logL_expon(cent, x, return_R=True)

        KSresErl = kstest(_r, lambda x: get_cdf(x,2,Rerl))
        KSresExp = kstest(_r, lambda x: get_cdf(x,1,Rexp))

        cdf = 1/N * np.arange(0,N)
        print(np.max(np.abs(cdf-get_cdf(_r,2,Rerl))))
        print(np.max(np.abs(cdf-get_cdf(_r,1,Rexp))))

        title = ['(0,0)','C_exp','C_erl','CoM'][i]

        print(title, f"{KSresErl=}")
        print(title, f"{KSresExp=}")


        a = ax[0,i]

        a.set_title(title)
        pdf, be, _ = a.hist(_r,bins=101,density=True)
        a.plot(rth, np.exp(-rth/Rexp)/Rexp, label=f'log(L)/N = {-Lexp:4.2f}')
        a.plot(rth, rth*np.exp(-rth/Rerl)/Rerl**2, label=f'log(L)/N = {-Lerl:4.2f}')

        a = ax[1,i]

        a.set_title(title)
        cdf =  np.cumsum(pdf*np.diff(be))
        print(cdf[0])
        a.step(be[1:],cdf)
        a.plot(rth, get_cdf(rth, 1, Rexp), label=f'log(L)/N = {-Lexp:4.2f}')
        a.plot(rth, get_cdf(rth, 2, Rerl), label=f'log(L)/N = {-Lerl:4.2f}')

        a.legend()

    pl.show()
