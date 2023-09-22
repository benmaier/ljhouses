import matplotlib.pyplot as pl
from ljhouses.fitcbdKS import *

if __name__ == "__main__":

    np.random.seed(1)

    N = 1000
    R = 1
    #pos = get_erlang_sample(N, R, phimin=np.pi*1.5)
    pos = get_erlang_sample(N, R, phimin=0)
    full_analysis(pos,extent=[-2,10,-10,2])


    pl.show()
