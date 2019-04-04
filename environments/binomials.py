# binomial_buchberger.py
# Dan Halpern-Leistner
# 7 Mar 2019

import numpy as np

# for now, the characteristic is fixed
CHAR = 32003

def grevlex(M1, M2):
    # returns 1 if M1 > M2 in the grevlex ordering, 0 if M1=M2, and -1 if M1<M2
    if np.sum(M1) > np.sum(M2):
        return 1
    elif np.sum(M1) < np.sum(M2):
        return -1
    else:
        for i in np.flip(M1-M2):
            if i > 0:
                return -1
            if i < 0:
                return 1
        return 0


def euclidean(a, b):
    if abs(b) > abs(a):
        (x,y,d) = euclidean(b, a)
        return (y,x,d)

    if abs(b) == 0:
        return (1, 0, a)

    x1, x2, y1, y2 = 0, 1, 1, 0
    while abs(b) > 0:
        q, r = divmod(a,b)
        x = x2 - q*x1
        y = y2 - q*y1
        a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y

    return (x2, y2, a)


def invert(a):
    x,y,d = euclidean(a % CHAR,CHAR)
    return x % CHAR


class Binomial(object):
    def __init__(self, a1, a2, M1, M2):
        """
        a1, a2 should be integers,
        M1, M2 should be integer numpy arrays of the same length
        """
        assert M1.shape == M2.shape
        self.M1 = M1
        self.M2 = M2
        self.a1 = a1
        self.a2 = a2
        self.simplify()

    def __str__(self):
        return '%i x^%s + %i x^%s' % (self.a1, self.M1, self.a2, self.M2)

    def __repr__(self):
        return '%i x^%s + %i x^%s' % (self.a1, self.M1, self.a2, self.M2)

    def simplify(self):
        """
        arranges that M1 >= M2 in grevlex order
        and rescales polynomial to be monic
        """
        self.a1 = self.a1 % CHAR
        if self.a1 == 0:
            self.M1 = np.zeros_like(self.M1)
        self.a2 = self.a2 % CHAR
        if self.a2 == 0:
            self.M2 = np.zeros_like(self.M2)

        order = grevlex(self.M1,self.M2)
        if order == -1:
            self.M1, self.M2 = self.M2, self.M1
            self.a1, self.a2 = self.a2, self.a1
        elif order == 0:
            self.M2 = np.zeros_like(self.M1)
            self.a1 = (self.a1+self.a2) % CHAR
            self.a2 = 0
            return

        if self.a1 is not 0:
            self.a2 = (self.a2 * invert(self.a1)) % CHAR
            self.a1 = 1


def spoly(f, g):
    """
    Return the s-polynomial of binomials f and g.
    Assumes both are simplified!
    """
    lcm = np.max([f.M1,g.M1],0)
    return Binomial(f.a2, (-g.a2) % CHAR, lcm - f.M1 + f.M2, lcm - g.M1 + g.M2)


def reduce_monomial(a, M, F):
    """Assumes polynomials in F are simplified, and nonzero"""
    r_a,r_M = a,M
    found_divisor = True
    while found_divisor:
        if r_a==0:
            return (r_a,r_M)
        found_divisor = False
        for f in F:
            if all(r_M >= f.M1):
                #print('(%i,%s)... using: %s' %(r_a,r_M,f))
                assert f.a1 is not 0, 'No polynomial in the set %s can be 0' %(F)
                r_a = (- r_a * f.a2) % CHAR
                r_M = r_M - f.M1 + f.M2
                found_divisor = True
                break
    #print('Monomial reduction complete')
    return (r_a,r_M)


def reduce(g, F):
    """
    Return the remainder when polynomial g is divided by polynomials F.
    Assumes g and the binomials in F are simplified!
    """
    # we reduce each monomial in g separately
    (a1,M1) = reduce_monomial(g.a1,g.M1,F)
    (a2,M2) = reduce_monomial(g.a2,g.M2,F)
    return Binomial(a1,a2,M1,M2) #this implicitly simplifies the result


def minimalize(G):
    """Return a minimal Groebner basis from arbitrary Groebner basis G."""
    Gmin = []
    for f in G:
        if all([not all(f.M1 >= g.M1) for g in Gmin]):
            Gmin = [g for g in Gmin if not all(g.M1 >= f.M1)]
            Gmin.append(f)
    return Gmin


def interreduce(G):
    """Return a list of the polynomials in G reduced with respect to each other."""
    Gred = []
    for i in range(len(G)):
        g = reduce(G[i], G[:i] + G[i+1:])
        Gred.append(g) #the output of reduce is automatically monic
    return Gred


def buchberger(F, verbose=False):
    """Return a Groebner basis from polynomials G using Buchberger's algorithm."""
    G = [g for g in F]
    P = [(i, j) for j in range(len(G)) for i in range(j)]

    while P:
        i, j = min(P)
        P.remove((i, j))

        s = spoly(G[i], G[j])
        if verbose:
            print('S polynomial: %s -->' %(s))
        r = reduce(s, G)

        if verbose and r.a1 == 0:
            print('zero reduction %s.\n' % (r))

        if r.a1 != 0:
            if verbose:
                print('added %s\n' % (r))
            j = len(G)
            for i in range(len(G)):
                P.append((i, j))
            G.append(r)

    G = minimalize(G)
    G = interreduce(G)

    return G


def random_partition(d, n):
    """Use bars and stars to get a random partition of integer d into n pieces."""
    bars = np.random.choice(d+n-1, n-1, replace=False)
    bars.sort()

    counts = np.empty(n, dtype=int)
    counts[0] = bars[0]
    for i in range(1, len(bars)):
        counts[i] = bars[i] - bars[i-1] - 1
    counts[n-1] = d + n - 1 - bars[-1] - 1

    return counts


def random_binomial(degree, num_variables, degree2=None):
    """Return a random binomial in variables in given degree."""
    if degree2 is None:
        degree2 = degree
    M1 = np.array(random_partition(degree, num_variables), dtype=int)
    M2 = np.array(random_partition(degree2, num_variables), dtype=int)
    while all(M2 == M1):
        M2 = np.array(random_partition(degree2, num_variables), dtype=int)
    a1 = np.random.randint(CHAR, dtype=np.int64)
    a2 = np.random.randint(CHAR, dtype=np.int64)
    return Binomial(a1,a2,M1,M2)


class BinomialBuchbergerEnv:
    """An environment for Groebner basis computation using Buchberger."""

    def __init__(self, degree, size, num_variables):
        self.num_variables = num_variables
        self.degree = degree
        self.size = size
        self.G = []
        self.P = []

    def step(self, action):
        """Perform one reduction and return the new polynomial list and pair list."""
        i, j = action
        self.P.remove((i, j))

        s = spoly(self.G[i], self.G[j])
        r = reduce(s, self.G)

        if r.a1 != 0:
            j = len(self.G)
            for i in range(len(self.G)):
                self.P.append((i, j))
            self.G.append(r)

        return (self.G, self.P), -1, len(self.P) == 0, {}

    def reset(self, G=None):
        """Initialize the polynomial list and pair list from polynomials G."""
        degrees = np.random.randint(1, self.degree + 1, size=(self.size, 2))
        if G is None:
            self.G = [random_binomial(degrees[i, 0], self.num_variables, degrees[i, 1]) for i in range(self.size)]
        else:
            self.G = G

        self.P = [(i, j) for j in range(len(self.G)) for i in range(j)]
        return self.G, self.P

    def render(self):
        print(self.P)
        print(self.G)
        print()


def monomial_tensor(state, k=1):
    """Return a (len(P), 1, 2*num_variables) tensor of pairs of lead monomials."""
    G, P = state
    if k == 1:
        vecs = [np.concatenate((G[p[0]].M1, G[p[1]].M1)) for p in P]
    else:
        vecs = [np.concatenate((G[p[0]].M1, G[p[0]].M2, G[p[1]].M1, G[p[1]].M2)) for p in P]
    return np.expand_dims(np.array(vecs, dtype=int), axis=1)


class LeadMonomialWrapper:
    """A wrapper for Buchberger environments that returns lead monomials as vectors."""

    def __init__(self, env, k=1):
        self.env = env
        self.state = None
        self.k = k

    def reset(self):
        self.state = self.env.reset()
        return monomial_tensor(self.state, k=self.k)

    def step(self, action):
        G, P = self.state
        action = P[action]
        self.state, reward, done, info = self.env.step(action)
        return monomial_tensor(self.state, k=self.k), reward, done, info
    
    
class DegreeAgent:

    def __init__(self, random=False):
        self.random = random

    def act(self, state):
        n = state.shape[2]//2  # number of variables
        degs = np.sum(np.maximum(state[:, :, :n], state[:, :, n:]), axis=2)
        if self.random:
            indices = np.where(degs == np.min(degs))[0]
            return np.random.choice(indices)
        else:
            return np.argmin(degs)


class RandomAgent:

    def act(self, state):
        return np.random.randint(state.shape[0])
