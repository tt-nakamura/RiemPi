import numpy as np
from mpmath import zetazero
from scipy.special import expi
from scipy.integrate import quad,cumtrapz
from sympy import mobius

ln2 = np.log(2)
g = lambda t: 1/(t**2-1)/t/np.log(t)

def RiemannJ(x, rho):
    """ return J(x) = sum_{n=1}^inf pi(x^{1/n})/n
    x = 1d array of real numbers (increasing order)
    rho = 1d array of complex zeros of zeta
    """
    u = np.log(x)
    tau = np.imag(rho)[:,np.newaxis]
    J = expi(u[0]) - ln2
    J += quad(g, x[0], np.inf)[0]
    J -= 2*np.sum(np.real(expi(rho*u[0])))
    y = x - 1/(x**2-1)
    y -= 2*np.sqrt(x)*np.sum(np.cos(tau*u), 0)
    J += cumtrapz(y/u, u, initial=0)
    return J

def RiemannPi(x1, x2, c=16, k=1, verbose=False):
    """ pi(x) (number of primes <= x) for x in [x1,x2]
    c controls the number of zeta zeros rho
    k controls the step size of integration by trapz
    use zeta zeros such that |Im(rho)| < c sqrt(x2)
    shortest wavelength is divided into 8*k trapezoids
    """
    T = c*np.sqrt(x2) # Im(rho) < T
    N = int(k*np.log(x2/x1)*T*4/np.pi) # number of points
    x = np.geomspace(x1,x2,N) # logarithmic interval
    rho = zeta_zeros(T, verbose) # defined below
    pi = RiemannJ(x,rho)
    for n in range(2, int(x2).bit_length()):
        mu = int(mobius(n)) # moebius inversion
        if mu==0: continue
        y = x**(1/n)
        pi[y>=2] += mu/n*RiemannJ(y[y>=2], rho)
    return x,pi

file_name = 'zeta_zeros.npy'

def zeta_zeros(T, verbose=False):
    """ return zeros of Riemann zeta function whose
    imaginary parts are less than T
    """
    try:
        rho = np.load(file_name)
        if np.imag(rho[-1]) >= T:
            return rho[np.imag(rho) < T]
        rho = rho.tolist()
    except IOError:
        rho = []

    z = complex(zetazero(len(rho)+1))
    while np.imag(z) < T:
        rho.append(z)
        if verbose: print('%d:'%len(rho), z)
        z = complex(zetazero(len(rho)+1))

    np.save(file_name, np.r_[rho, z])
    return np.asarray(rho)
