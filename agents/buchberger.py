# buchberger.py
# Dylan Peifer
# 26 Feb 2019
"""Agents for Buchberger environments."""

import numpy as np
import sympy as sp


def lm(f, order):
    return sp.poly(sp.LM(f, order=order), *f.gens, domain=f.domain)


class FirstAgent:
    """Agent which chooses the first pair on the list."""

    def act(self, state):
        _, P = state
        return P[0]


class RandomAgent:
    """Agent which chooses a random pair on the list."""

    def act(self, state):
        _, P = state
        return P[np.random.randint(len(P))]


def pair_degree(pair, G, order):
    """Return the degree of the lcm of the lead monomials of the pair."""
    f, g = G[pair[0]], G[pair[1]]
    lcm = sp.lcm(lm(f, order), lm(g, order))
    return sum(sp.degree_list(lcm))


class DegreeAgent:
    """Agent which chooses the first pair with minimal degree of lcm."""

    def __init__(self, order):
        self.order = order

    def act(self, state):
        G, P = state
        return min(P, key=lambda pair: pair_degree(pair, G, self.order))


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


class NormalAgent:
    """Agent which chooses the first pair with minimal lcm in the monomial order."""

    def __init__(self, order):
        self.order = order

    def act(self, state):
        G, P = state
        return min(P, key=lambda pair: pair_normal_tuple(pair, G, self.order))
