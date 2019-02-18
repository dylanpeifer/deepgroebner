# buchberger.py
# Dylan Peifer
# 18 Feb 2019
"""Agents for Buchberger environments."""

import numpy as np
import sympy as sp


def lm(f, order):
    return sp.poly(sp.LM(f, order=order), *f.gens, domain=f.domain)


class FirstAgent:

    def act(self, state):
        _, P = state
        return min(P)


class RandomAgent:

    def act(self, state):
        _, P = state
        index = np.random.choice(len(P))
        return list(P)[index]


def degree_normal_tuple(pair, G):
    """Return a tuple which gives ordering of pair in degree normal selection."""
    f, g = G[pair[0]], G[pair[1]]
    lcm = sp.lcm(lm(f, 'grevlex'), lm(g, 'grevlex'))
    return sum(sp.degree_list(lcm))


class DegreeAgent:

    def act(self, state):
        G, P = state
        return min(P, key=lambda pair: degree_normal_tuple(pair, G))


def grevlex_normal_tuple(pair, G):
    """Return a tuple which gives ordering of pair in grevlex normal selection."""
    f, g = G[pair[0]], G[pair[1]]
    lcm = sp.lcm(lm(f, 'grevlex'), lm(g, 'grevlex'))
    vec = sp.degree_list(lcm)
    return sum(vec), tuple(reversed([-x for x in vec]))


class NormalAgent:

    def act(self, state):
        G, P = state
        return min(P, key=lambda pair: grevlex_normal_tuple(pair, G))
