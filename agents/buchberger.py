# buchberger.py
# Dylan Peifer
# 31 Mar 2019
"""Agents for Buchberger environments."""

import numpy as np
import sympy as sp


def lm(f, order):
    """Return lead monomial of polynomial f as a polynomial in the same ring."""
    return sp.poly(sp.LM(f, order=order), *f.gens, domain=f.domain)


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
        
    def act(self, state):
        pass
