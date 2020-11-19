"""Tests for Buchberger environments."""

import pytest
import sympy as sp

from deepgroebner.buchberger import *
from deepgroebner.ideals import FixedIdealGenerator


R1, x, y, z = sp.ring('x,y,z', sp.FF(32003), 'grevlex')
R2, a, b, c, d = sp.ring('a,b,c,d', sp.QQ, 'lex')
R3, t, u, v = sp.ring('t,u,v', sp.FF(101), 'grlex')


@pytest.mark.parametrize("f, g, s", [
    (x**2 + x*y, y**2 + x*y, 0),
    (x**3*y**2 - x**2*y**3, x**4*y + y**2, -x**3*y**3 - y**3),
    (x**2 + y**3, x*y**2 + x + 1, x**3 - x*y - y),
    (a**2 + a*b, b**2 + a*b, 0),
    (a**3*b**2 - a**2*b**3, a**4*b + b**2, -a**3*b**3 - b**3),
    (a**2 - b**3, a*b**2 + a + 1, -b**5 - a**2 - a),
    (t**2 + t*u, u**2 + t*u, 0),
    (t**3*u**2 - t**2*u**3, t**4*u + u**2, -t**3*u**3 - u**3),
    (t**2 + u**3, t*u**2 + t + 1, t**3 - t*u - u),
])
def test_spoly(f, g, s):
    assert spoly(f, g) == s


@pytest.mark.parametrize("g, F, r, s", [
    (x**5*y**10*z**4 + 22982*x**3*y*z**2,
     [x**5*y**12 + 25797*x*y**5*z**2, x*y**3*z + 27630*x**2*y, x**2*y**9*z + 8749*x**2],
     2065*x**9*y**2 + 22982*x**3*y*z**2,
     4),
    (a**5*c + a**3*b + a**2*b**2 + a*b**2 + a,
     [a**2*c - a, a*b**2 + c**5, a*c + c**3/4],
     a**4 + a**3*b + a + c**7/4 - c**5,
     4),
    (a**3*b*c**2 + a**2*c,
     [a**2 + b, a*b*c + c, a*c**2 + b**2],
     b*c**2 - b*c,
     3),
])
def test_reduce(g, F, r, s):
    assert reduce(g, F) == (r, {'steps': s})


@pytest.mark.parametrize("s, p", [
    ('degree', (0, 1)), ('normal', (0, 1)), ('first', (0, 1)),
])
def test_select_0(s, p):
    G = [x**2 + y, x*y + x, z**3 + x + y]
    P = set([(0, 1), (0, 2), (1, 2)])
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    (['degree', 'first'], (0, 2)), ('normal', (1, 2)), ('first', (0, 1)),
])
def test_select_1(s, p):
    G = [x*y + 1, z**2 + x + z, y*z + x]
    P = set([(0, 1), (0, 2), (1, 2)])
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    ('normal', (0, 2)), ('first', (0, 2)), ('random', (0, 2)),
])
def test_select_2(s, p):
    G = [x*y + 1, z**2 + x + z, y*z + x]
    P = set([(0, 2)])
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    (['degree', 'first'], (0, 1)),
    (['degree', 'normal'], (1, 3)),
    ('normal', (1, 2)),
])
def test_select_3(s, p):
    G = [a*b + c*d**3, c*d + d, d**5, c**2*d**2]
    P = set([(0, 1), (1, 2), (1, 3)])
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    ('first', (0, 2)), ('normal', (1, 2)),
    (['degree', 'first'], (1, 3)),
    (['degree', 'normal'], (1, 4)),
])
def test_select_4(s, p):
    G = [a*b*c, c*d, d**5, a*b, c**2*d**2]
    P = set([(0, 2), (1, 2), (1, 3), (1, 4)])
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("s, p", [
    ('first', (1, 2)),
    (['first', 'random'], (1, 2)),
    ('normal', (0, 3)),
    (['degree', 'first'], (0, 3)),
    (['degree', 'normal', 'first'], (0, 3)),
])
def test_select_5(s, p):
    G = [t*u**2 + t**2, u*v + 1, v**5 + t, u**3 + t*u]
    P = set([(0, 3), (1, 2)])
    assert select(G, P, strategy=s) == p


@pytest.mark.parametrize("f", [x**2 + x*y + 2, a*b + a])
@pytest.mark.parametrize("s", ['none', 'lcm', 'gebauermoeller'])
def test_update_0(f, s):
    assert update([], set(), f, strategy=s) == ([f], set())


@pytest.mark.parametrize("s, P_", [
    ('none', set([(0, 1)])), ('lcm', set()), ('gebauermoeller', set()),
])
def test_update_1(s, P_):
    G = [x*y**2 + 2*x*z - x]
    f = z**5 + 2*x**2*y*z + x*z
    assert update(G, set(), f, strategy=s) == (G + [f] , P_)


@pytest.mark.parametrize("s, P_", [
    ('none', set([(0, 1), (0, 2), (1, 2)])),
    ('lcm', set([(0, 1), (0, 2), (1, 2)])),
    ('gebauermoeller', set([(0, 2), (1, 2)])),
])
def test_update_2(s, P_):
    G = [a*b**2 + 2*c, a*c**2 - b**2 - c]
    f = a + b**2*c + 4*c**2 + 1
    assert update(G, set([(0, 1)]), f, strategy=s) == (G + [f], P_)


@pytest.mark.parametrize("s, P_", [
    ('none', set([(0, 1), (0, 2), (1, 2)])),
    ('lcm', set([(0, 1), (1, 2)])),
    ('gebauermoeller', set([(0, 1), (1, 2)])),
])
def test_update_3(s, P_):
    G = [a*b**2 + 2*c, a*c**2 - b**2 - c]
    f = 4*c**2 + 1
    assert update(G, set([(0, 1)]), f, strategy=s) == (G + [f], P_)


@pytest.mark.parametrize("s, P_", [
    ('none', set([(0, 1), (0, 2), (1, 2)])),
    ('lcm', set([(0, 1), (0, 2), (1, 2)])),
    ('gebauermoeller', set([(0, 1), (0, 2)])),
])
def test_update_4(s, P_):
    G = [a*b**2 + 2*c, a*c**2 - b**2 - c]
    f = 4*b**2*c + b*c**2
    assert update(G, set([(0, 1)]), f, strategy=s) == (G + [f], P_)


@pytest.mark.parametrize("s, P_", [
    ('none', set([(0, 2), (0, 3), (1, 3), (2, 3)])),
    ('lcm', set([(0, 2), (0, 3), (1, 3)])),
    ('gebauermoeller', set([(0, 2)])),
])
def test_update_5(s, P_):
    G = [x*y**2 + 2*z, x*z**2 - y**2 - z, x + 3]
    f = y**2*z**3 - y**2 + 4*z**4 + z**2
    assert update(G, set([(0, 2)]), f, strategy=s) == (G + [f], P_)


@pytest.mark.parametrize("s, P_", [
    ('none', set([(0, 4), (1, 4), (2, 4), (3, 4)])),
    ('lcm', set([(0, 4), (1, 4), (3, 4)])),
    ('gebauermoeller', set([(3, 4)])),
])
def test_update_5(s, P_):
    G = [a*b**2 + 2*c, a*c**2 - b**2 - c, -a + b**2*c + 4*c**2, b**2*c**3 - b**2 + 4]
    f = b**4*c + 4*b**2*c**2 + b**2 + 2*c
    assert update(G, set(), f, strategy=s) == (G + [f], P_)


@pytest.mark.parametrize("s, P_", [
    ('none', set((i, 5) for i in range(5))),
    ('lcm', set([(0, 5), (3, 5), (4, 5)])),
    ('gebauermoeller', set([(4, 5)])),
])
def test_update_6(s, P_):
    G = [a*b**2 + 2*c, a*c**2 - b**2, -a + b**2*c, b**2*c**3 - b**2, b**4*c + 4*b**2]
    f = -b**4 - b**2*c - 2*c**3 - c**2/2
    assert update(G, set(), f, strategy=s) == (G + [f], P_)


def test_update_7():
    G = [a*b**2 + 2*c, a*c**2 - b**2, -a + b**2*c, b**2*c**3 - b**2, b**4*c + 4*b**2, b**4 - b**2]
    f = b**2*c**2 + b**2 - c**4 - c**3
    G_, P_ = update(G, set(), f, strategy='gebauermoeller')
    assert P_ == set([(3, 6), (4, 6)]) or P_ == set([(3, 6), (5, 6)])


def test_update_8():
    G = [a*b**2, a*c**2, -a, b**2*c**3, b**4*c, b**4, b**2*c**2]
    f = b**2*c + 14*b**2 - 8*c**5 - 58*c**4 + c**2 + c
    G_, P_ = update(G, set(), f, strategy='gebauermoeller')
    assert P_ == set([(4, 7), (6, 7)]) or P_ == set([(5, 7), (6, 7)])


@pytest.mark.parametrize("G, Gmin", [
    ([], []),
    ([x*y**2 + z, x*z + 3*y, x**2 + y*z, -3*y**3 + z**2, -3*y - z**3/3, z**8/243 + z],
     [x*z + 3*y, x**2 + y*z, -z**3/3 - 3*y, -3*y**3 + z**2, x*y**2 + z]),
    ([a*b**2 + c, a*c + 3*b, a**2 + b*c, -3*b**3 + c**2, -3*b - c**3/3, c**8/243 + c],
     [c**8/243 + c, -3*b - c**3/3, a*c + 3*b, a**2 + b*c]),
])
def test_minimalize(G, Gmin):
    assert minimalize(G) == Gmin


@pytest.mark.parametrize("G, Gred", [
    ([], []),
    ([x*z + 3*y, x**2 + y*z, -z**3/3 - 3*y, -3*y**3 + z**2, x*y**2 + z],
     [x*z + 3*y, x**2 + y*z, z**3 + 9*y, y**3 - z**2/3, x*y**2 + z]),
    ([c**8/243 + c, -3*b - c**3/3, a*c + 3*b, a**2 + b*c],
     [c**8 + 243*c, b + c**3/9, a*c - c**3/3, a**2 - c**4/9]),
])
def test_interreduce(G, Gred):
    assert interreduce(G) == Gred


@pytest.mark.parametrize("F, G", [
    ([], []),
    ([y - x**2, z - x**3], [y**2 - x*z, x*y - z, x**2 - y]),
    ([b - a**2, c - a**3], [b**3 - c**2, a*c - b**2, a*b - c, a**2 - b]),
    ([u - t**2, v - t**3], [t*v - u**2, t*u - v, t**2 - u, u**3 - v**2]),
    ([x + y + z, x*y + y*z + x*z, x*y*z - 1], [x + y + z, y**2 + y*z + z**2, z**3 - 1]),
]) 
@pytest.mark.parametrize("s", ['first', 'degree', 'normal', 'random'])
@pytest.mark.parametrize("e", ['none', 'lcm', 'gebauermoeller'])
def test_buchberger(F, G, s, e):
    assert buchberger(F, selection=s, elimination=e) == G


@pytest.mark.parametrize("sort_reducers, r", [
    (True, b**2*c**3 + b*c*d - c**2),
    (False, b**2*c**3 + b**2*c**2 + b*d - c**2),
])
def test_BuchbergerEnv_0(sort_reducers, r):
    F = [a**2*b*d - c**2, a*d - b*c**2 - d, a - c]
    ideal_gen = FixedIdealGenerator(F)
    env = BuchbergerEnv(ideal_gen, sort_reducers=sort_reducers)
    env.reset()
    (G, P), _, _, _ = env.step((0, 1))
    assert len(G) == 4 and G[3] == r


def run_episode(agent, env):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


@pytest.mark.parametrize("s", ['first', ['degree', 'first'], ['normal', 'first']])
def test_episode_0(s):
    R, a, b, c, d, e = sp.ring('a,b,c,d,e', sp.FF(32003), 'grevlex')
    F = [a + 2*b + 2*c + 2*d + 2*e - 1,
         a**2 + 2*b**2 + 2*c**2 + 2*d**2 + 2*e**2 - a,
         2*a*b + 2*b*c + 2*c*d + 2*d*e - b,
         b**2 + 2*a*c + 2*b*d + 2*c*e - c,
         2*b*c + 2*a*d + 2*b*e - d]
    ideal_gen = FixedIdealGenerator(F)
    env = BuchbergerEnv(ideal_gen, rewards='reductions')
    agent = BuchbergerAgent(selection=s)
    assert run_episode(agent, env) == -28


@pytest.mark.parametrize("e, reward", [
    ('none', -45), ('lcm', -35), ('gebauermoeller', -11),
])
def test_episode_1(e, reward):
    R, a, b, c, d = sp.ring('a,b,c,d', sp.FF(32003), 'grevlex')
    F = [a + b + c + d,
         a*b + b*c + c*d + d*a,
         a*b*c + b*c*d + c*d*a + d*a*b,
         a*b*c*d - 1]
    ideal_gen = FixedIdealGenerator(F)
    env = BuchbergerEnv(ideal_gen, elimination=e, rewards='reductions')
    agent = BuchbergerAgent(selection=['normal', 'first'])
    assert run_episode(agent, env) == reward


@pytest.mark.parametrize("s, reward", [
    ('first', -49), (['degree', 'first'], -57), (['normal', 'first'], -63),
])
def test_episode_2(s, reward):
    R, x, y, z, t = sp.ring('x,y,z,t', sp.FF(32003), 'grlex')
    F = [x**31 - x**6 - x - y, x**8 - z, x**10 - t]
    ideal_gen = FixedIdealGenerator(F)
    env = BuchbergerEnv(ideal_gen, rewards='reductions')
    agent = BuchbergerAgent(selection=s)
    assert run_episode(agent, env) == reward


@pytest.mark.parametrize("g, k, v", [
    (R1.one, 1, [0, 0, 0]),
    (R2.zero, 2, [0, 0, 0, 0, 0, 0, 0, 0]),
    (x*y, 1, [1, 1, 0]),
    (x*y, 3, [1, 1, 0, 0, 0, 0, 0, 0, 0]),
    (x*y**2*z + x**3 + z + 1, 1, [1, 2, 1]),
    (x*y**2*z + x**3 + z + 1, 2, [1, 2, 1, 3, 0, 0]),
    (x*y**2*z + x**3 + z + 1, 4, [1, 2, 1, 3, 0, 0, 0, 0, 1, 0, 0, 0]),
    (b*d**5 + a**3, 1, [3, 0, 0, 0]),
    (b*d**5 + a**3, 3, [3, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0]),
    (u**3*v + t**2, 1, [0, 3, 1]),
    (u**3*v + t**2, 2, [0, 3, 1, 2, 0, 0]),
])
def test_lead_monomials_vector(g, k, v):
    assert np.array_equal(lead_monomials_vector(g, k=k), np.array(v))


def test_LeadMonomialsEnv_0():
    R, x, y, z = sp.ring('x,y,z', sp.FF(101), 'grevlex')
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    env = LeadMonomialsEnv(ideal_gen, elimination='none')
    state = env.reset()
    assert np.array_equal(state, np.array([[2, 0, 0, 3, 0, 0]]))
    state, _, done, _ = env.step(0)
    assert (np.array_equal(state, np.array([[2, 0, 0, 1, 1, 0], [3, 0, 0, 1, 1, 0]])) or
            np.array_equal(state, np.array([[3, 0, 0, 1, 1, 0], [2, 0, 0, 1, 1, 0]])))
    assert not done
    action = 0 if np.array_equal(state[0], np.array([3, 0, 0, 1, 1, 0])) else 1
    state, _, done, _ = env.step(action)
    assert np.array_equal(state, np.array([[2, 0, 0, 1, 1, 0]]))
    assert not done
    for _ in range(4):
        state, _, done, _ = env.step(0)
    assert done


def test_LeadMonomialsEnv_1():
    R, x, y, z = sp.ring('x,y,z', sp.FF(101), 'grevlex')
    F = [y - x**2, z - x**3]
    ideal_gen = FixedIdealGenerator(F)
    env = LeadMonomialsEnv(ideal_gen)
    state = env.reset()
    assert np.array_equal(state, np.array([[2, 0, 0, 3, 0, 0]]))
    state, _, done, _ = env.step(0)
    assert np.array_equal(state, np.array([[2, 0, 0, 1, 1, 0]]))
    assert not done
    state, _, done, _ = env.step(0)
    assert np.array_equal(state, np.array([[1, 1, 0, 0, 2, 0]]))
    assert not done
    state, _, done, _ = env.step(0)
    assert done


@pytest.mark.parametrize("selection, k, action", [
    ('degree', 1, 2),
    ('degree', 2, 1),
    ('first', 1, 0),
    ('first', 2, 0),
])
def test_LeadMonomialsAgent(selection, k, action):
    agent = LeadMonomialsAgent(selection=selection, k=k)
    state = np.array([[11,  1,  2,  7,  2,  5,  5, 12,  2,  0,  1,  2],
                      [ 1, 17,  0,  1,  5, 10,  0, 16,  3,  1, 10,  7],
                      [ 0,  8,  7,  9,  0,  2,  5, 12,  2,  0,  1,  2],
                      [ 0,  8,  7,  9,  0,  2,  0, 16,  3,  1, 10,  7],
                      [11,  1,  2,  7,  2,  5,  0,  0, 12,  9,  0,  2]])
    assert agent.act(state) == action
