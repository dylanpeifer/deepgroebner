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


def reduce(g, F, lmF=None):
    """Return remainder when g is divided by monic polynomials F."""
    ring = g.ring
    monomial_div = ring.monomial_div
    lmF = [f.LM for f in F] if lmF is None else lmF

    stats = {'steps': 0}
    r = ring.zero
    g = g.copy()

    while g:
        lmg, lcg = g.LT
        found_divisor = False
        
        for f, lmf in zip(F, lmF):
            m = monomial_div(lmg, lmf)
            if m is not None:
                g = g - f.mul_term((m, lcg))
                found_divisor = True
                stats['steps'] += 1
                break

        if not found_divisor:
            if lmg in r:
                r[lmg] += lcg
            else:
                r[lmg] = lcg
            del g[lmg]

    return r, stats


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
    ideal_gen
        A generator which yields ideals as lists of polynomials.
    elimination : {'gebauermoeller', 'lcm', 'none'}, optional
        The elimination strategy used when updating the pair set.
    sort_reducers : bool, optional
        Whether to choose reducers in sorted order when performing long division
        on the s-polynomials.
    rewards : {'reductions', 'additions'}, optional
        The reward value for each step.

    Examples
    --------
    >>> import sympy as sp
    >>> R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    >>> ideal_gen = FixedIdealGenerator([y - x**2, z - x**3])
    >>> env = BuchbergerEnv(ideal_gen)
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
                 ideal_gen,
                 elimination='gebauermoeller',
                 sort_reducers=True,
                 rewards='additions'):
        self.ideal_gen = ideal_gen
        self.elimination = elimination
        self.sort_reducers = sort_reducers
        self.rewards = rewards
        self.ring = None
        self.G = []
        self.P = set()
        self.reducers = []
        self.lmReducers = []

    def reset(self):
        """Initialize the polynomial list and pair list for a new ideal from ideal_fn."""
        F = next(self.ideal_gen)
        self.ring = F[0].ring
        self.G = []
        self.lmG = []
        self.P = set()
        for f in F:
            self.G, self.P = update(self.G, self.P, f.monic(), lmG=self.lmG, strategy=self.elimination)
            self.lmG.append(f.LM)
        if self.sort_reducers:
            self.reducers = sorted([f.monic() for f in F], key=lambda f: self.ring.order(f.LM))
        else:
            self.reducers = [f.monic() for f in F]
        self.lmReducers = [f.LM for f in self.reducers]
        return (self.G, self.P) if self.P else self.reset()

    def step(self, action):
        """Perform one reduction and return the new polynomial list and pair list."""
        i, j = action
        self.P.remove((i, j))
        s = spoly(self.G[i], self.G[j], lmf=self.lmG[i], lmg=self.lmG[j])
        r, stats = reduce(s, self.reducers, lmF=self.lmReducers)
        if r != 0:
            self.G, self.P = update(self.G, self.P, r.monic(), lmG=self.lmG, strategy=self.elimination)
            self.lmG.append(r.LM)
            if self.sort_reducers:
                self.reducers = sorted(self.reducers + [r.monic()], key=lambda f: self.ring.order(f.LM))
                self.lmReducers = [f.LM for f in self.reducers]
            else:
                self.reducers.append(r.monic())
                self.lmReducers.append(r.LM)
        reward = -(1 + stats['steps']) if self.rewards == 'additions' else -1
        return (self.G, self.P), reward, len(self.P) == 0, {}

    def render(self):
        print(self.G)
        print(self.P)
        print()

    def copy(self):
        """Return a copy of this environment with the same state."""
        copy = BuchbergerEnv(self.ideal_gen)
        copy.elimination = self.elimination
        copy.sort_reducers = self.sort_reducers
        copy.rewards = self.rewards
        copy.ring = self.ring
        copy.G = [g.copy() for g in self.G]
        copy.lmG = self.lmG[:]
        copy.P = self.P.copy()
        copy.reducers = [r.copy() for r in self.reducers]
        return copy

    @property
    def actions(self):
        return list(self.P)


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


def lead_monomials_vector(g, k=1, dtype=np.int):
    """Return the concatenated exponent vectors of the k lead monomials of g."""
    n = g.ring.ngens
    it = iter(g.monoms())
    return np.array([next(it, (0,) * n) for _ in range(k)]).flatten().astype(dtype)


class LeadMonomialsWrapper:
    """A wrapper for BuchbergerEnv with state a matrix of the pairs' lead monomials.
    
    Parameters
    ----------
    env : BuchbergerEnv
        The environment that will be wrapped.
    k : int, optional
        The number of lead monomials used for each polynomial.
    dtype : data-type, optional
        The data-type used for the state matrix.
    
    Examples
    --------
    >>> import sympy as sp
    >>> R, x, y, z = sp.ring("x,y,z", sp.FF(32003), 'grevlex')
    >>> ideal_gen = FixedIdealGenerator([y - x**2, z - x**3])
    >>> env = BuchbergerEnv(ideal_gen)
    >>> wrapped_env = LeadMonomialsWrapper(env, k=2)
    >>> wrapped_env.reset()
    array([[2, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 1]])
    >>> wrapped_env.step(0)
    (array([[2, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1]]), -1, False, {})
    
    """

    def __init__(self, env, k=1, dtype=np.int):
        self.env = env
        self.k = k
        self.dtype = dtype
        self.pairs = []       # list of current pairs
        self.m = 0            # size of current basis
        self.leads = {}       # leads[i] = lead_monomials_vector(env.G[i])

    def reset(self):
        G, P = self.env.reset()
        self.pairs = list(P)
        self.m = len(G)
        self.leads = {i: lead_monomials_vector(G[i], k=self.k, dtype=self.dtype)
                      for i in range(self.m)}
        return self._matrix()

    def step(self, action):
        (G, P), reward, done, info = self.env.step(self.pairs[action])
        self.pairs = list(P)
        if len(G) > self.m:
            self.m += 1
            self.leads[self.m-1] = lead_monomials_vector(G[self.m-1], k=self.k, dtype=self.dtype)
        return self._matrix(), reward, done, info

    def render(self):
        self.env.render()

    def copy(self):
        env = self.env.copy()
        copy = LeadMonomialsWrapper(env, k=self.k, dtype=self.dtype)
        copy.pairs = self.pairs.copy()
        copy.m = self.m
        copy.leads = self.leads.copy()
        return copy

    def _matrix(self):
        if self.pairs:
            return np.array([np.concatenate([self.leads[p[0]], self.leads[p[1]]])
                                             for p in self.pairs])
        else:
            n = self.env.G[0].ring.ngens
            return np.zeros((0, 2*n*self.k), dtype=self.dtype)


class LeadMonomialsAgent:
    """An agent that follows standard selection strategies.

    Parameters
    ----------
    selection : {'first', 'degree', 'random'}
        The selection strategy used to pick pairs.
    """

    def __init__(self, selection='degree', k=1):
        self.strategy = selection
        self.k = k

    def act(self, state):
        if self.strategy == 'first':
            return 0
        elif self.strategy == 'degree':
            n = state.shape[1] // (2 * self.k)
            m = state.shape[1] // 2
            return np.argmin(np.sum(np.maximum(state[:, :n], state[:, m:m+n]), axis=1))
        elif self.strategy == 'random':
            return np.random.choice(len(state))
        elif self.strategy == 'normal':
            n = state.shape[1] // (2 * self.k)
            m = state.shape[1] // 2
            all_lcm = np.maximum(state[:, :n], state[:, m:m+n])
            min_lcm = sorted(all_lcm, key = lambda i:[n for n in i])[0]
            for i, lcm in enumerate(all_lcm):
                if np.sum(min_lcm == lcm) == n:
                    return i

            

