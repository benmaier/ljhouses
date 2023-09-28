from ljhouses.pairwisetheory import square
import numpy as np
import matplotlib.pyplot as pl




#pl.figure()
#pl.plot(x, pdf)



import numpy as np
import matplotlib.pyplot as pl

from ljhouses.pythonsims import simulate_collisions_until_no_collisions_simple
from ljhouses.drawing import plot_configuration

from fincoretails import unipareto

from HousesInCountries.analyses import histogram



alpha = 3.33
xmin = 0.3

N = 1400
radiuses = unipareto.sample(N, alpha, xmin)

L = 1
R = L/np.sqrt(N)/np.pi * radiuses
R0 = 0.0001
R0 = 0.002
R += R0
x = np.random.rand(N,2) * L

#pop = (R/R0)**2
pop = np.ones_like(R)


fig, ax = pl.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
ax[0].set_title('initial configuration')

plot_configuration(x, R, ax[0])
xnew = simulate_collisions_until_no_collisions_simple(x, R,mass_prop_to_area=True)
plot_configuration(xnew, R, ax[1])
ax[1].set_title('after running collision detection')

fig.tight_layout()
#fig.savefig('collision_example_varying_radius.png', dpi=150)

from scipy.spatial.distance import pdist

dists = pdist(xnew)

npairs = np.zeros_like(dists)
for i in range(N-1):
    for j in range(i+1,N):
        ndx = N * i + j - ((i + 2) * (i + 1)) // 2
        npairs[ndx] = pop[i]*pop[j]


if R0 == 0:
    rlog = np.logspace(-3,np.log10(np.sqrt(2)),51)
else:
    rlog = np.logspace(np.log10(R0)-0.5,np.log10(np.sqrt(2)),51)

r = np.linspace(0,0.05,51)
rmid = 0.5*(r[1:]+r[:-1])
rlmid = 0.5*(rlog[1:]+rlog[:-1])
new_counts_log, _ = histogram(dists, npairs, rlog)
new_counts_lin, _ = histogram(dists, npairs, r)

pdflog = square(rlmid,1)
pdf = rmid / (r[-1]**2/2)

pl.figure()
pl.plot(rlmid, new_counts_log)
pl.plot(rlmid, pdflog)
#pl.plot(rmid, pdf)
#pl.plot(rmid, new_counts_lin)
pl.xscale('log')
pl.yscale('log')

pl.figure()
pl.plot(rmid, new_counts_lin/pdf)
#pl.xscale('log')
pl.ylim([0,2])
#pl.yscale('log')



pl.show()
