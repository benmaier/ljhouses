import numpy as np
from scipy.special import gamma
from scipy.optimize import curve_fit

def erlang(x,R):
    return x*np.exp(-x/R)/R**2

def erlang_R_from_mean(mean):
    return mean/2

def erlang_from_mean(x, mean):
    R = erlang_R_from_mean(mean)
    return erlang(x,R)

def half_gauss(x,R):
    return 2*x*np.exp(-(x/R)**2)/R**2

def half_gauss_R_from_mean(mean):
    return mean*2/np.sqrt(np.pi)

def half_gauss_from_mean(x,mean):
    R = half_gauss_R_from_mean(mean)
    return half_gauss(x,R)

def generalized_pairwise(x,R,m):
    assert(m > 0)
    return m / gamma(2/m) * x / R**2 * np.exp(-(x/R)**m)

def generalized_pairwise_R_from_mean(mean, m):
    R = gamma(2/m) * mean / gamma(3/m)
    return R

def generalized_pairwise_from_mean(x, mean, m):
    R = generalized_pairwise_R_from_mean(mean, m)
    return generalized_pairwise(x, R, m)

def circle(x,R):
    if hasattr(R,'__len__'):
        i0 = np.where(2*R<x)[0]
        i1 = np.where(2*R>=x)[0]
        y0 = np.zeros_like(i0,dtype=float)
        y1 = 4 * x/np.pi/R[i1]**2 * np.arccos(x/2/R[i1]) - 2*x**2/np.pi/R[i1]**3 * np.sqrt(1-x**2/4/R[i1]**2)
        y = np.concatenate((y0,y1))
    elif hasattr(x,'__len__'):
        i0 = np.where(2*R<x)[0]
        i1 = np.where(2*R>=x)[0]
        y0 = np.zeros_like(i0,dtype=float)
        y1 = 4 * x[i1]/np.pi/R**2 * np.arccos(x[i1]/2/R) - 2*x[i1]**2/np.pi/R**3 * np.sqrt(1-x[i1]**2/4/R**2)
        y = np.concatenate((y1,y0))
    else:
        y = 4 * x/np.pi/R**2 * np.arccos(x/2/R) - 2*x**2/np.pi/R**3 * np.sqrt(1-x**2/4/R**2)
    return y

def square(x, L):

    is_x_arr = hasattr(x, '__len__')
    if not is_x_arr:
        x = np.array([x])

    r = x / L
    s = r**2

    y = np.zeros_like(x)
    i0 = np.where(s<=1)[0]
    i1 = np.where(np.logical_and(s>1, s<=2))[0]

    if len(i0) > 0:
        y0 = -4*r[i0] + np.pi + s[i0]
        y[i0] = y0
    if len(i1) > 0:
        y1 = -2 - np.pi -s[i1] + 4*np.arcsin(1/r[i1]) + 4*(s[i1]-1)**0.5
        y[i1] = y1

    f = 2*r*y
    if not is_x_arr:
        return f[0]
    else:
        return f


def circle_from_mean(x, mean):
    R = circle_R_from_mean(mean)
    return circle(x, R)

def circle_R_from_mean(mean):
    return mean * 45 * np.pi / 128

def parabola(x,R):
    if hasattr(R,'__len__'):
        i0 = np.where(2*R<x)[0]
        i1 = np.where(2*R>=x)[0]
        y0 = np.zeros_like(i0,dtype=float)
        y1 = -0.75 * (x/R[i1])**2/R[i1] + 1.5 * (x/R[i1]**2)
        y = np.concatenate((y0,y1))
    elif hasattr(x,'__len__'):
        i0 = np.where(2*R<x)[0]
        i1 = np.where(2*R>=x)[0]
        y0 = np.zeros_like(i0,dtype=float)
        y1 = -0.75 * (x[i1]/R)**2/R + 1.5 * (x[i1]/R**2)
        y = np.concatenate((y1,y0))
    else:
        y1 = -0.75 * (x/R)**2/R + 1.5 * (x/R**2)
    return y

def parabola_R_from_mean(mean):
    return mean

def parabola_from_mean(x, mean):
    R = parabola_R_from_mean(mean)
    return parabola(x, R)

def fit_generalized_pairwise(xdata,ydata,m0=1.5,log=False):
    """Returns estimated R, m for the generalized model from curve fit"""
    mean = ydata.dot(xdata)*(xdata[1]-xdata[0])
    if log:
        ndx = np.where(ydata>0)[0]
        xdata = xdata[ndx]
        ydata = np.log(ydata[ndx])
        func = lambda x, R, m: np.log(generalized_pairwise(x, R, m))
    else:
        func = generalized_pairwise

    m = m0
    R = generalized_pairwise_R_from_mean(mean, m)
    popt, pcov = curve_fit(func, xdata, ydata, p0=(R,m))

    return popt

if __name__=="__main__":
    import matplotlib.pyplot as pl
    rs0 = np.linspace(0,2,1001)
    rs1 = np.linspace(0,5,1001)

    fig, ax = pl.subplots(2,2,figsize=(8,8))

    for a in ax[0,:]:
        a.plot(rs0, circle(rs0, 1), label='Circle')
        a.plot(rs1, erlang(rs1, 1), label='Erlang')
        a.plot(rs0, parabola(rs0, 1), label='Parabola')
        a.plot(rs1, half_gauss(rs1, 1), label='Half Gauss')

        m = 1.5
        a.plot(rs1, generalized_pairwise(rs1, 1,m), label=f'Generalized, m={m:4.2f}')



    for a in ax[1,:]:
        a.plot(rs1, circle_from_mean(rs1, 1), label='Circle')
        a.plot(rs1, erlang_from_mean(rs1, 1), label='Erlang')
        a.plot(rs1, parabola_from_mean(rs1, 1), label='Parabola')
        a.plot(rs1, half_gauss_from_mean(rs1, 1), label='Half Gauss')

        m = 1.5
        a.plot(rs1, generalized_pairwise_from_mean(rs1, 1,m), label=f'Generalized, m={m:4.2f}')

    for a in ax.flatten():
        a.set_xlabel('pairwise r/R')
        a.set_ylabel('pdf',loc='top')
        a.legend()
    for a in ax[:,1]:
        a.set_yscale('log')
        a.set_ylim(1e-3,a.get_ylim()[1])
    fig.tight_layout()


    pl.show()
