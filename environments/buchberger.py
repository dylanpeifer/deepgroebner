# buchberger.py
# Dylan Peifer
# 18 Jun 2019
"""An environment for computing Groebner bases with Buchberger's algorithm."""

import numpy as np
import sympy as sp


def spoly(f, g, lmf=None, lmg=None):
    """Return the s-polynomial of monic polynomials f and g."""
    assert f.ring == g.ring, "polynomials must be in same ring"
    lmf = f.LM if lmf is None else lmf
    lmg = g.LM if lmg is None else lmg
    R = f.ring
    lcm = R.monomial_lcm(lmf, lmg)
    s1 = f.mul_monom(R.monomial_div(lcm, lmf))
    s2 = g.mul_monom(R.monomial_div(lcm, lmg))
    return s1 - s2


def reduce(g, F):
    """Return the remainder when polynomial g is divided by polynomials F."""
    # TODO: return dictionary of statistics
    return g.rem(F), {}


def select(G, P, strategy='normal'):
    """Select and return a pair from P."""
    assert len(G) > 0, "polynomial list must be nonempty"
    assert len(P) > 0, "pair set must be nonempty"
    R = G[0].ring

    if isinstance(strategy, str):
        strategy = [strategy]

    def strategy_key(p, s):
        """Return a sort key for pair p in the strategy s."""
        if s == 'first':
            return p[1], p[0]
        elif s == 'normal':
            lcm = R.monomial_lcm(G[p[0]].LM, G[p[1]].LM)
            return R.order(lcm)
        elif s == 'degree':
            lcm = R.monomial_lcm(G[p[0]].LM, G[p[1]].LM)
            return sum(lcm)
        elif s == 'random':
            return np.random.rand()
        else:
            raise ValueError('unknown selection strategy')

    return min(P, key=lambda p: tuple(strategy_key(p, s) for s in strategy))


def update(G, P, f, lmG=None, strategy='gebauermoeller'):
    """Return the new list of polynomials and set of pairs when f is added to the basis G."""
    lmf = f.LM
    lmG = [g.LM for g in G] if lmG is None else lmG
    R = f.ring
    lcm = R.monomial_lcm
    mul = R.monomial_mul
    div = R.monomial_div

    if strategy == 'none':
        P_ = {(i, len(G)) for i in range(len(G))}
    elif strategy == 'lcm':
        P_ = {(i, len(G)) for i in range(len(G)) if lcm(lmG[i], lmf) != mul(lmG[i], lmf)}
    elif strategy == 'gebauermoeller':
        P = {p for p in P if (not div(lcm(lmG[p[0]], lmG[p[1]]), lmf) or
                              lcm(lmG[p[0]], lmG[p[1]]) == lcm(lmG[p[0]], lmf) or
                              lcm(lmG[p[0]], lmG[p[1]]) == lcm(lmG[p[1]], lmf))}
        lcm_dict = {}
        for i in range(len(G)):
            lcm_dict.setdefault(lcm(lmG[i], lmf), []).append(i)
        minimalized_lcms = []
        for L in sorted(lcm_dict.keys(), key=R.order):
            if all(not div(L, L_) for L_ in minimalized_lcms):
                minimalized_lcms.append(L)
        P_ = set()
        for L in minimalized_lcms:
            if not any(lcm(lmG[i], lmf) == mul(lmG[i], lmf) for i in lcm_dict[L]):
                P_.add((min(lcm_dict[L]), len(G)))
    else:
        raise ValueError('unknown elimination strategy')

    return G + [f], P | P_


def minimalize(G):
    """Return a minimal Groebner basis from arbitrary Groebner basis G."""
    R = G[0].ring if len(G) > 0 else None
    assert all(g.ring == R for g in G), "polynomials must be in same ring"    
    Gmin = []
    for f in sorted(G, key=lambda h: R.order(h.LM)):
        if all(not R.monomial_div(f.LM, g.LM) for g in Gmin):
            Gmin.append(f)
    return Gmin


def interreduce(G):
    """Return the reduced Groebner basis from a minimal Groebner basis G."""
    R = G[0].ring if len(G) > 0 else None
    assert all(g.ring == R for g in G), "polynomials must be in same ring"
    Gred = []
    for i in range(len(G)):
        g = G[i].rem(G[:i] + G[i+1:])
        Gred.append(g.monic())
    return Gred


def buchberger(F, selection='normal', elimination='gebauermoeller'):
    """Return a Groebner basis from polynomials F using Buchberger's algorithm."""
    R = F[0].ring if len(F) > 0 else None
    assert all(f.ring == R for f in F), "polynomials must be in same ring"

    G = []
    lmG = []
    P = set()
    for f in F:
        G, P = update(G, P, f.monic(), strategy=elimination)
        lmG.append(f.LM)

    while P:
        i, j = select(G, P, strategy=selection)
        P.remove((i, j))
        s = spoly(G[i], G[j], lmf=lmG[i], lmg=lmG[j])
        r, _ = reduce(s, G)
        if r != 0:
            G, P = update(G, P, r.monic(), lmG=lmG, strategy=elimination)
            lmG.append(r.LM)

    return interreduce(minimalize(G))


class BuchbergerEnv:
    """An environment for computing a Groebner basis using Buchberger's algorithm.
    
    Parameters
    ----------
    ideal_fn
        A function with signature ring -> list that generates ideals.
    ring
        The polynomial ring which will contain the generated ideals. 
    elimination : {'gebauermoeller', 'lcm', 'none'}, optional
        The elimination strategy used when updating the pair set.
    sort_reducers : bool, optional
        Whether to choose reducers in sorted order when performing long division
        on the s-polynomials.
        
    Examples
    --------
    >>> import sympy as sp
    >>> R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    >>> ideal_fn = lambda R: [y - x**2, z - x**3]
    >>> env = BuchbergerEnv(ideal_fn)
    >>> env.reset()
    ([x**2 + 32002 mod 32003*y, x**3 + 32002 mod 32003*z], {(0, 1)})
    >>> env.step((0, 1))
    (([x**2 + 32002 mod 32003*y,
       x**3 + 32002 mod 32003*z,
       x*y + 32002 mod 32003*z],
      {(0, 2)}),
     -1,
     False,
     {})
    
    """

    def __init__(self,
                 ideal_fn,
                 ring=sp.xring('x,y,z', sp.FF(32003), 'grevlex')[0],
                 elimination='gebauermoeller',
                 sort_reducers=False):
        self.ideal_fn = ideal_fn
        self.ring = ring
        self.elimination = elimination
        self.sort_reducers = sort_reducers
        self.G = []
        self.P = set()
        self.reducers = []

    def reset(self):
        """Initialize the polynomial list and pair list for a new ideal from ideal_fn."""
        F = self.ideal_fn(self.ring)
        self.G = []
        self.lmG = []
        self.P = set()
        for f in F:
            self.G, self.P = update(self.G, self.P, f.monic(), lmG=self.lmG, strategy=self.elimination)
            self.lmG.append(f.LM)
        self.reducers = sorted(F, key=lambda f: self.ring.order(f.LM)) if self.sort_reducers else F
        return (self.G, self.P) if self.P else self.reset()

    def step(self, action):
        """Perform one reduction and return the new polynomial list and pair list."""
        i, j = action
        self.P.remove((i, j))
        s = spoly(self.G[i], self.G[j], lmf=self.lmG[i], lmg=self.lmG[j])
        r, _ = reduce(s, self.reducers)
        if r != 0:
            self.G, self.P = update(self.G, self.P, r.monic(), lmG=self.lmG, strategy=self.elimination)
            self.lmG.append(r.LM)
            if self.sort_reducers:
                self.reducers = sorted(self.reducers + [r], key=lambda f: self.ring.order(f.LM))
            else:
                self.reducers.append(r)
        return (self.G, self.P), -1, len(self.P) == 0, {}

    def render(self):
        print(self.G)
        print(self.P)
        print()


class BuchbergerAgent:
    """An agent that follows standard selection strategies.
    
    Parameters
    ----------
    selection : {'normal', 'first', 'degree', 'random'}
        The selection strategy used to pick pairs.
    """
    
    def __init__(self, selection='normal'):
        self.strategy = selection
        
    def act(self, state):
        G, P = state
        return select(G, P, strategy=self.strategy)


def lead_monomials_vector(g, k=1):
    """Return the concatenated exponent vectors of the k lead monomials of g."""
    n = g.ring.ngens
    it = iter(g.monoms())
    return np.array([next(it, (0,) * n) for _ in range(k)]).flatten()


class LeadMonomialsWrapper():
    """A wrapper for BuchbergerEnv with state a matrix of the pairs' lead monomials."""

    def __init__(self, env, k=1):
        self.env = env
        self.k = k
        self.pairs = []       # list of current pairs
        self.m = 0            # size of current basis
        self.leads = {}       # leads[i] = lead_monomials_vector(env.G[i])
        self.pair_leads = {}  # pair_leads[(i, j)] = np.concatenate([leads[i], leads[j]])

    def reset(self):
        G, P = self.env.reset()
        self.pairs = list(P)
        self.m = len(G)
        self.leads = {i: lead_monomials_vector(G[i], k=self.k) for i in range(self.m)}
        self.pair_leads = {(i, j): np.concatenate([self.leads[i], self.leads[j]])
                           for i in range(self.m) for j in range(i+1, self.m)}
        return self._matrix()

    def step(self, action):
        (G, P), reward, done, info = self.env.step(self.pairs[action])
        self.pairs = list(P)
        if len(G) > self.m:
            self.m += 1
            self.leads[self.m-1] = lead_monomials_vector(G[self.m-1], k=self.k)
            new_pairs = {(i, self.m-1): np.concatenate([self.leads[i], self.leads[self.m-1]])
                         for i in range(self.m-1)}
            self.pair_leads.update(new_pairs)
        return self._matrix(), reward, done, info

    def render(self):
        self.env.render()

    def _matrix(self):
        if self.pairs:
            return np.array([self.pair_leads[p] for p in self.pairs])
        else:
            n = self.env.G[0].ring.ngens
            return np.zeros((0, 2*n*self.k))
