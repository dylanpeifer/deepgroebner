# buchberger.py
# Dylan Peifer
# 26 Feb 2019
"""An environment for computing Groebner bases with Buchberger."""

import numpy as np
import sympy as sp


def lm(f, order):
    """Return lead monomial of polynomial f as a polynomial in the same ring."""
    return sp.poly(sp.LM(f, order=order), *f.gens, domain=f.domain)


def lt(f, order):
    """Return lead term of polynomial f as a polynomial in the same ring."""
    return sp.poly(sp.LT(f, order=order), *f.gens, domain=f.domain)


def spoly(f, g, order):
    """Return the s-polynomial of polynomials f and g."""
    lcm = sp.lcm(lm(f, order), lm(g, order))
    return lcm//lt(f, order) * f - lcm//lt(g, order) * g


def reduce(g, F, order):
    """Return the remainder when polynomial g is divided by polynomials F."""
    r = g.zero()
    while g != 0:

        found_divisor = False

        for f in F:
            if lm(g, order) % lm(f, order) == 0:
                g -= lt(g, order)//lt(f, order) * f
                found_divisor = True
                break

        if not found_divisor:
            lg = lt(g, order)
            r += lg
            g -= lg

    return r


def minimalize(G, order):
    """Return a minimal Groebner basis from arbitrary Groebner basis G."""
    Gmin = []
    for f in G:
        if all([lm(f, order) % lm(g, order) != 0 for g in Gmin]):
            Gmin = [g for g in Gmin if lm(g, order) % lm(f, order) != 0]
            Gmin.append(f)
    return Gmin


def interreduce(G, order):
    """Return a list of the polynomials in G reduced with respect to each other."""
    Gred = []
    for i in range(len(G)):
        g = reduce(G[i], G[:i] + G[i+1:], order)
        Gred.append(g.quo_ground(sp.LC(g, order=order)))
    return Gred


def buchberger(F, variables, domain, order):
    """Return a Groebner basis from polynomials F using Buchberger's algorithm."""
    G = [sp.poly(f, *variables, domain=domain) for f in F]
    P = [(i, j) for j in range(len(G)) for i in range(j)]

    while P:
        i, j = min(P, key=lambda p: (p[1], p[0]))  # first selection
        P.remove((i, j))

        s = spoly(G[i], G[j], order)
        r = reduce(s, G, order)

        if r != 0:
            j = len(G)
            for i in range(len(G)):
                P.append((i, j))
            G.append(r)

    G = minimalize(G, order)
    G = interreduce(G, order)

    return G


class BuchbergerEnv:
    """An environment for Groebner basis computation using Buchberger."""

    def __init__(self, variables, domain=sp.FF(32003), order='grevlex', f=None):
        self.variables = variables
        self.domain = domain
        self.order = order
        self.G = []
        self.P = []
        self.f = f

    def step(self, action):
        """Perform one reduction and return the new polynomial list and pair list."""
        i, j = action
        self.P.remove((i, j))

        s = spoly(self.G[i], self.G[j], self.order)
        r = reduce(s, self.G, self.order)

        if r != 0:
            j = len(self.G)
            for i in range(len(self.G)):
                self.P.append((i, j))
            self.G.append(r)

        return (self.G, self.P), -1, len(self.P) == 0, {}

    def reset(self, F=None):
        """Initialize the polynomial list and pair list from polynomials F."""
        self.G = [sp.poly(f, *self.variables, domain=self.domain) for f in F]
        self.P = [(i, j) for j in range(len(self.G)) for i in range(j)]
        return self.G, self.P

    def render(self):
        print(self.G)
        print(self.P)
        print()


def monomial_tensor(state, order):
    """Return a (len(P), 1, 2*len(variables)) tensor of pairs of lead monomials."""
    G, P = state
    vecs = [sp.degree_list(lm(G[p[0]], order)) + sp.degree_list(lm(G[p[1]], order)) for p in P]
    return np.expand_dims(np.array(vecs, dtype=int), axis=1)


class LeadMonomialWrapper:
    """A wrapper for Buchberger environments that returns lead monomials as vectors."""
    
    def __init__(self, env):
        self.env = env
        self.state = None
        
    def reset(self):
        self.state = self.env.reset()
        return monomial_tensor(self.state, self.env.order)
    
    def step(self, action):
        G, P = self.state
        action = P[action]
        self.state, reward, done, info = self.env.step(action)
        return monomial_tensor(self.state, self.env.order), reward, done, info


def random_partition(n, k):
    """Use bars and stars to get a random partition of integer n into k pieces."""
    bars = np.random.choice(n+k-1, k-1, replace=False)
    bars.sort()

    counts = np.empty(k, dtype=int)
    counts[0] = bars[0]
    for i in range(1, len(bars)):
        counts[i] = bars[i] - bars[i-1] - 1
    counts[k-1] = n + k - 1 - bars[-1] - 1

    return counts


def exponent_to_monomial(exponent, variables):
    """Convert an exponent vector (as a list) into a monomial in variables."""
    return np.product([x**k for x, k in zip(variables, exponent)])


def random_binomial(degree, variables):
    """Return a random binomial in variables in given degree."""
    n = len(variables)
    m1 = exponent_to_monomial(random_partition(degree, n), variables)
    m2 = exponent_to_monomial(random_partition(degree, n), variables)
    while m2 == m1:
        m2 = exponent_to_monomial(random_partition(degree, n), variables)
    return m1 + m2
