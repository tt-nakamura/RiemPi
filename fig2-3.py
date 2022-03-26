import numpy as np
import matplotlib.pyplot as plt
from sympy import primepi
from RiemannPi import RiemannPi

K = [16,32]

# fig2 #########################################################
x1,x2 = 2,50
xe = range(x1,x2)
ye = [primepi(x) for x in xe]

plt.figure(figsize=(5, 5.4))

for i,k in enumerate(K):
    x,y = RiemannPi(x1, x2, k)
    plt.subplot(len(K),1,i+1)
    plt.plot(xe, ye, 'r.', label='Eratosthenes')
    plt.plot(x, y, 'b', label='Riemann')
    plt.plot(x, np.round(y), 'g--', label='Riemann (rounded)')
    plt.ylabel(r'$\pi(x)$')
    plt.yticks([0,5,10,15])
    plt.text(1,14,'k=%d'%k)
    plt.legend(loc='lower right')

plt.xlabel(r'$x$')
plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()

# fig3 #########################################################
x1,x2 = 100,150
xe = range(x1,x2)
ye = [primepi(x) for x in xe]

plt.figure(figsize=(5, 5.4))

for i,k in enumerate(K):
    x,y = RiemannPi(x1, x2, k)
    plt.subplot(len(K),1,i+1)
    plt.plot(xe, ye, 'r.', label='Eratosthenes')
    plt.plot(x, y, 'b', label='Riemann')
    plt.plot(x, np.round(y), 'g--', label='Riemann (rounded)')
    plt.ylabel(r'$\pi(x)$')
    plt.yticks([25,30,35])
    plt.text(99,34.2,'k=%d'%k)
    plt.legend(loc='lower right')

plt.xlabel(r'$x$')
plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()
