# buchberger.py
# Dylan Peifer
# 01 Apr 2019
"""An environment for computing Groebner bases with Buchberger."""

import numpy as np
import sympy as sp


def lm(f, order):
    """Return lead monomial of polynomial f as a polynomial in the same ring."""
    return sp.poly(sp.LM(f, order=order), *f.gens, domain=f.domain)


def lt(f, order):
    """Return lead term of polynomial f as a polynomial in the same ring."""
    return sp.poly(sp.LT(f, order=order), *f.gens, domain=f.domain)


def lcm(f, g, order):
    """Return the lcm of the lead monomials of f and g."""
    return sp.lcm(lm(f, order), lm(g, order))


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


def update_pairs(G, P, f, order, strategy='gebauermoeller'):
    """Return the updated list of polynomials and pairs when f is added to the basis G."""
    G.append(f)
    P_ = [(i, len(G)-1) for i in range(len(G)-1)]
    
    if strategy == 'none':
        return G, P + P_
    elif strategy == 'lcm':
        P_ = [p_ for p_ in P_ if sp.gcd(lm(G[p_[0]], order), lm(G[p_[1]], order)) != 1]
        return G, P + P_
    elif strategy == 'gebauermoeller':
        P = [p for p in P if any(lcm(G[p[0]], G[p[1]], order) % lm(f, order) != 0,
                                 lcm(G[p[0]], G[p[1]], order) == lcm(G[p[0]], lm(f, order), order),
                                 lcm(G[p[0]], G[p[1]], order) == lcm(G[p[0]], lm(f, order), order))]
        P_ = [p_ for p_ in P_ if all(lcm(G[p_[0]], G[p_[1]], order) % lcm(G[p[0]], G[p[1]], order) != 0 or
                                     lcm(G[p_[0]], G[p_[1]], order) == lcm(G[p[0]], G[p[1]], order)
                                     for p in P_)]
        lcms = {lcm(G[p[0]], G[p[1]], order) : p for p in P_}
        P_ = list(lcms.values())
        P_ = [p_ for p_ in P_ if sp.gcd(lm(G[p_[0]], order), lm(G[p_[1]], order)) != 1]
        return G, P + P_


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
            G, P = update_pairs(G, P, r, order)

    G = minimalize(G, order)
    G = interreduce(G, order)

    return G


class BuchbergerEnv:
    """An environment for Groebner basis computation using Buchberger."""

    def __init__(self, variables, domain=sp.FF(32003), order='grevlex', elimination='gebauermoeller'):
        self.variables = variables
        self.domain = domain
        self.order = order
        self.elimination = elimination
        self.G = []
        self.P = []

    def step(self, action):
        """Perform one reduction and return the new polynomial list and pair list."""
        i, j = action
        self.P.remove((i, j))

        s = spoly(self.G[i], self.G[j], self.order)
        r = reduce(s, self.G, self.order)

        if r != 0:
            self.G, self.P = update_pairs(self.G, self.P, r, self.order, strategy=self.elimination)

        return (self.G, self.P), -1, len(self.P) == 0, {}

    def reset(self, F=None):
        """Initialize the polynomial list and pair list from polynomials F."""
        if F is None:
            F = self.f(self.variables, self.domain)
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
    
    
def pair_degree(pair, G, order):
    """Return the degree of the lcm of the lead monomials of the pair."""
    f, g = G[pair[0]], G[pair[1]]
    lcm = sp.lcm(lm(f, order), lm(g, order))
    return sum(sp.degree_list(lcm))


def pair_normal_tuple(pair, G, order):
    """Return a tuple which gives ordering of lcm of pair in given order."""
    f, g = G[pair[0]], G[pair[1]]
    lcm = sp.lcm(lm(f, order), lm(g, order))
    vec = sp.degree_list(lcm)
    if order == 'lex':
        return vec
    elif order == 'grlex':
        return sum(vec), vec
    elif order == 'grevlex':
        return sum(vec), tuple(reversed([-x for x in vec]))


class BuchbergerAgent:
    
    def __init__(self, strategy='normal', order='grevlex'):
        self.strategy = strategy
        self.order = order
        
    def act(self, state):
        G, P = state
        if self.strategy == 'first':
            return min(P, key=lambda p: (p[1], p[0]))
        elif self.strategy == 'random':
            return P[np.random.randint(len(P))]
        elif self.strategy == 'degree':
            return min(P, key=lambda p: pair_degree(p, G, self.order))
        elif self.strategy == 'normal':
            return min(P, key=lambda p: pair_normal_tuple(p, G, self.order))


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
