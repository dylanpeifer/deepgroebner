# buchberger.py
# Dylan Peifer
# 18 Feb 2019
"""An environment for computing Groebner bases with Buchberger."""

import numpy as np
import sympy as sp


x, y, z = sp.symbols('x y z')
variables = [x, y, z]


def lm(f, order):
    return sp.poly(sp.LM(f, order=order), *f.gens, domain=f.domain)


def lt(f, order):
    return sp.poly(sp.LT(f, order=order), *f.gens, domain=f.domain)


def spoly(f, g, order):
    lcm = sp.lcm(lm(f, order), lm(g, order))
    return lcm//lt(f, order) * f - lcm//lt(g, order) * g


def reduce(g, F, order):
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
    Gmin = []
    for f in G:
        if all([lm(f, order) % lm(g, order) != 0 for g in Gmin]):
            Gmin = [g for g in Gmin if lm(g, order) % lm(f, order) != 0]
            Gmin.append(f)
    return Gmin


def interreduce(G, order):
    Gred = []
    for i in range(len(G)):
        g = reduce(G[i], G[:i] + G[i+1:], order)
        Gred.append(g.quo_ground(sp.LC(g, order=order)))
    return Gred


def buchberger(F, variables, domain, order):
    G = [sp.poly(f, *variables, domain=domain) for f in F]
    P = set((i, j) for j in range(len(G)) for i in range(j))

    while P:
        i, j = min(P)
        P.remove((i, j))

        s = spoly(G[i], G[j], order)
        r = reduce(s, G, order)

        if r != 0:
            j = len(G)
            for i in range(len(G)):
                P.add((i, j))
            G.append(r)

    G = minimalize(G, order)
    G = interreduce(G, order)

    return G


class BuchbergerEnv:
    """An environment for Groebner basis computation using Buchberger."""

    def __init__(self, domain=sp.FF(32003), order='grevlex'):
        self.G = []
        self.P = set()
        self.domain = domain
        self.order = order

    def step(self, action):
        i, j = action
        self.P.remove((i, j))

        s = spoly(self.G[i], self.G[j], self.order)
        r = reduce(s, self.G, self.order)

        if r != 0:
            j = len(self.G)
            for i in range(len(self.G)):
                self.P.add((i, j))
            self.G.append(r)

        return (self.G, self.P), -1, len(self.P) == 0, {}

    def reset(self, F, variables):
        self.G = [sp.poly(f, *variables, domain=self.domain) for f in F]
        self.P = set((i, j) for j in range(len(self.G)) for i in range(j))
        return self.G, self.P

    def render(self):
        print(self.G)
        print(self.P)
        print()


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


def exponent_to_monomial(exponent):
    """Convert an exponent 3-vector (as a list) into a monomial in x,y,z."""
    return np.product([x**k for x, k in zip(variables, exponent)])


def random_binomial(degree):
    """Return a random binomial in x,y,z in given degree."""
    m1 = exponent_to_monomial(random_partition(degree, 3))
    m2 = exponent_to_monomial(random_partition(degree, 3))
    while m2 == m1:
        m2 = exponent_to_monomial(random_partition(degree, 3))
    return m1 + m2


class BinomialBuchbergerEnv(BuchbergerEnv):

    def __init__(self, degree, size):
        BuchbergerEnv.__init__(self)
        self.degree = degree
        self.size = size

    def reset(self):
        F = [random_binomial(self.degree) for _ in range(self.size)]
        return BuchbergerEnv.reset(self, F, variables)


def state_to_tensor(state):
    G, P = state
    vecs = []
    for pair in P:
        vec = sp.degree_list(lm(G[pair[0]], 'grevlex')) + sp.degree_list(lm(G[pair[1]], 'grevlex'))
        vecs.append(vec)
    return np.expand_dims(np.array(vecs), axis=1)


class LeadMonomialWrapper:
    """A wrapper for Buchberger environments that returns lead monomials as vectors."""
    
    def __init__(self, env):
        self.env = env
        self.state = None
        
    def reset(self):
        self.state = self.env.reset()
        return state_to_tensor(self.state)
    
    def step(self, action):
        G, P = self.state
        action = list(P)[action]
        self.state, reward, done, info = self.env.step(action)
        return state_to_tensor(self.state), reward, done, info
