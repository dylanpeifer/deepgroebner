/*
 * An environment for computing Groebner bases with Buchberger's algorithm.
 */

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "buchberger.h"
#include "ideals.h"
#include "polynomials.h"


Polynomial spoly(const Polynomial& f, const Polynomial& g) {
  Term gamma = {1, lcm(f.LM(), g.LM())};
  return (gamma / f.LT()) * f - (gamma / g.LT()) * g;
}


std::pair<Polynomial, ReduceStats> reduce(const Polynomial& g, const std::vector<Polynomial>& F) {
  int steps = 0;
  Polynomial r;
  Polynomial h = g;

  while (h.size() != 0) {
    bool found_divisor = false;

    for (const Polynomial& f : F) {
      if (is_divisible(h.LM(), f.LM())) {
	h = h - (h.LT() / f.LT()) * f;
	found_divisor = true;
	steps++;
	break;
      }
    }

    if (!found_divisor) {
      r = r + Polynomial{h.LT()};
      h = h - Polynomial{h.LT()};
    }

  }

  return {r + h, {steps}};
}


void update(std::vector<Polynomial>& G, std::vector<SPair>& P, const Polynomial& f, EliminationType elimination) {

  int m = G.size();
  std::vector<SPair> P_;
  
  switch (elimination) {
  case EliminationType::None:
    for (int i = 0; i < m; i++) {
      P_.push_back(SPair{i, m});
    }
    break;
  case EliminationType::LCM:
    for (int i = 0; i < m; i++) {
      if (lcm(G[i].LM(), f.LM()) != G[i].LM() * f.LM())
	P_.push_back(SPair{i, m});
    }
    break;
  case EliminationType::GebauerMoeller:
    auto can_drop = [&G, &f](const SPair& p) {
		      Monomial m = lcm(G[p.i].LM(), G[p.j].LM());
		      return (is_divisible(m, f.LM()) &&
			      (m != lcm(G[p.i].LM(), f.LM())) &&
			      (m != lcm(G[p.j].LM(), f.LM())));
	            };
    P.erase(std::remove_if(P.begin(), P.end(), can_drop), P.end());

    std::map<Monomial, std::vector<int>> lcms;
    for (int i = 0; i < m; i++) {
      lcms[lcm(G[i].LM(), f.LM())].push_back(i);
    }
    std::vector<Monomial> min_lcms;
    for (const auto& pair : lcms) {  // will be in sorted order because std::map
      Monomial mon = pair.first;
      std::vector<int> v = pair.second;
      if (std::all_of(min_lcms.begin(), min_lcms.end(), [&mon](const Monomial& m) { return !is_divisible(mon, m); })) {
	min_lcms.push_back(mon);
	if (std::none_of(v.begin(), v.end(), [&G, &f](int i) { return lcm(G[i].LM(), f.LM()) == G[i].LM() * f.LM(); }))
	  P_.push_back(SPair{v[0], m});
      }
    }
    std::sort(P_.begin(), P_.end(), [](const SPair& p1, const SPair& p2) { return p1.i < p2.i; });

    break;
  }

  G.push_back(f);
  P.insert(P.end(), P_.begin(), P_.end());
}


std::vector<Polynomial> minimalize(const std::vector<Polynomial>& G) {
  std::vector<Polynomial> G_ = G;
  std::sort(G_.begin(), G_.end(), [](const Polynomial& f, const Polynomial& g) { return f.LM() < g.LM(); });
  std::vector<Polynomial> Gmin;
  for (const Polynomial& g : G_) {
    if (std::none_of(Gmin.begin(), Gmin.end(), [&g](const Polynomial& f) { return is_divisible(g.LM(), f.LM()); })) {
      Gmin.push_back(g);
    }
  }
  return Gmin;
}


std::vector<Polynomial> interreduce(const std::vector<Polynomial>& G) {
  std::vector<Polynomial> Gred;
  for (const Polynomial& g : G) {
    Term t = {1 / g.LC(), {}};
    Gred.push_back(t * (reduce(g - Polynomial{g.LT()}, G).first + Polynomial{g.LT()}));
  }
  return Gred;
}


std::pair<std::vector<Polynomial>, BuchbergerStats> buchberger(const std::vector<Polynomial>& F,
							       EliminationType elimination,
							       RewardType rewards,
							       bool sort_input,
							       bool sort_reducers,
							       double gamma) {
  std::vector<Polynomial> G;
  std::vector<SPair> P;
  for (const Polynomial& f : F) {
    update(G, P, f, elimination);
  }

  return buchberger(G, P, elimination, rewards, sort_reducers, gamma);
}


std::pair<std::vector<Polynomial>, BuchbergerStats> buchberger(const std::vector<Polynomial>& F,
							       const std::vector<SPair>& S,
							       EliminationType elimination,
							       RewardType rewards,
							       bool sort_reducers,
							       double gamma) {
  std::vector<Polynomial> G = F;
  std::vector<Polynomial> G_ = F;
  std::vector<SPair> P = S;
  BuchbergerStats stats = {};
  double discount = 1.0;

  if (sort_reducers)
    std::sort(G_.begin(), G_.end(), [](const Polynomial& f, const Polynomial& g) { return f.LM() < g.LM(); });

  auto first = [](const SPair& p1, const SPair& p2) {
		 return (p1.j < p2.j) || (p1.j == p2.j && p1.i < p2.i);
	       };

  auto degree = [&G](const SPair& p1, const SPair& p2) {
		  Monomial m1 = lcm(G[p1.i].LM(), G[p1.j].LM());
		  Monomial m2 = lcm(G[p2.i].LM(), G[p2.j].LM());
		  return m1.deg() < m2.deg();
		};

  auto normal = [&G](const SPair& p1, const SPair& p2) {
		  Monomial m1 = lcm(G[p1.i].LM(), G[p1.j].LM());
		  Monomial m2 = lcm(G[p2.i].LM(), G[p2.j].LM());
		  return m1 < m2;
		};

  auto sugar = [&G](const SPair& p1, const SPair& p2) {
		  Monomial m1 = lcm(G[p1.i].LM(), G[p1.j].LM());
		  Monomial m2 = lcm(G[p2.i].LM(), G[p2.j].LM());
		  int s1 = std::max(G[p1.i].sugar() + (m1 / G[p1.i].LM()).deg(),
				    G[p1.j].sugar() + (m1 / G[p1.j].LM()).deg());
		  int s2 = std::max(G[p2.i].sugar() + (m2 / G[p2.i].LM()).deg(),
				    G[p2.j].sugar() + (m2 / G[p2.j].LM()).deg());
		  return (s1 < s2) || (s1 == s2 && m1 < m2);
	       };

  while (!P.empty()) {
    auto iter = std::min_element(P.begin(), P.end(), degree);
    SPair p = *iter;
    P.erase(iter);
    auto [r, s] = reduce(spoly(G[p.i], G[p.j]), G_);
    double reward = (rewards == RewardType::Additions) ? (-1.0 - s.steps) : -1.0;
    stats.polynomial_additions += s.steps + 1;
    stats.total_reward += reward;
    stats.discounted_return += discount * reward;
    discount *= gamma;
    if (r.size() != 0) {
      update(G, P, r, elimination);
      stats.nonzero_reductions++;
      if (sort_reducers)
	G_.insert(std::upper_bound(G_.begin(), G_.end(), r, [](const Polynomial& f, const Polynomial& g) { return f.LM() < g.LM(); }), r);
      else
	G_.push_back(r);
    } else {
      stats.zero_reductions++;
    }
  }

  return {interreduce(minimalize(G)), stats};
}


BuchbergerEnv::BuchbergerEnv(std::string ideal_dist,
			     EliminationType elimination,
			     RewardType rewards,
			     bool sort_input,
			     bool sort_reducers)
    : elimination(elimination), rewards(rewards), sort_input(sort_input), sort_reducers(sort_reducers) {
  ideal_gen = parse_ideal_dist(ideal_dist);
}


BuchbergerEnv::BuchbergerEnv(const BuchbergerEnv& other)
    : ideal_gen(other.ideal_gen->copy()),
      G(other.G), P(other.P), elimination(other.elimination), rewards(other.rewards),
      sort_input(other.sort_input), sort_reducers(other.sort_reducers), G_(other.G_) {
}


BuchbergerEnv& BuchbergerEnv::operator=(const BuchbergerEnv& other) {
    // the only non-default copy is ideal_gen
    ideal_gen = other.ideal_gen->copy();
    G = other.G;
    P = other.P;
    elimination = other.elimination;
    rewards = other.rewards;
    sort_input = other.sort_input;
    sort_reducers = other.sort_reducers;
    G_ = other.G_;
    return *this;
}

void BuchbergerEnv::reset() {
  std::vector<Polynomial> F = ideal_gen->next();
  if (sort_input)
    std::sort(F.begin(), F.end(), [](const Polynomial& f, const Polynomial& g) { return f.LM() < g.LM(); });
  G.clear();
  G_.clear();
  P.clear();
  for (const Polynomial& f : F) {
    update(G, P, f, elimination);
    if (sort_reducers)
      G_.insert(std::upper_bound(G_.begin(), G_.end(), f, [](const Polynomial& f, const Polynomial& g) { return f.LM() < g.LM(); }), f);
    else
      G_.push_back(f);
  }
  if (P.empty())
    reset();
}


double BuchbergerEnv::step(SPair action) {
  P.erase(std::remove(P.begin(), P.end(), action), P.end());
  auto [r, stats] = reduce(spoly(G[action.i], G[action.j]), G_);
  if (r.size() != 0) {
    update(G, P, r, elimination);
    if (sort_reducers)
      G_.insert(std::upper_bound(G_.begin(), G_.end(), r, [](const Polynomial& f, const Polynomial& g) { return f.LM() < g.LM(); }), r);
    else
      G_.push_back(r);
  }
  return (rewards == RewardType::Additions) ? (-1.0 - stats.steps) : -1.0;
}


double BuchbergerEnv::value(double gamma) const {
  auto [G_, stats] = buchberger(G, P, elimination, rewards, sort_reducers, gamma);
  return stats.discounted_return;
}


std::vector<int> lead_monomials_vector(const Polynomial& f, int k, int n) {
  std::vector<int> lead;
  int i = 0;
  for (const Term& t : f.terms) {
    for (int j = 0; j < n; j++) {
      lead.push_back(t.monom[j]);
    }
    if (++i == k) break;
  }
  while (i < k) {
    for (int j = 0; j < n; j++) {
      lead.push_back(0);
    }
    i++;
  }
  return lead;
}


LeadMonomialsEnv::LeadMonomialsEnv(std::string ideal_dist,
				   bool sort_input,
				   bool sort_reducers,
				   int k)
  : env{ideal_dist, EliminationType::GebauerMoeller, RewardType::Additions, sort_input, sort_reducers}, k(k)
{
  n = env.nvars();
  cols = 2 * env.nvars() * k;
}


void LeadMonomialsEnv::reset() {
  env.reset();
  state.clear();
  leads.clear();
  for (const Polynomial& g : env.G) {
    leads.push_back(lead_monomials_vector(g, k, n));
  }
  for (const auto& p : env.P) {
    state.insert(state.end(), leads[p.i].begin(), leads[p.i].end());
    state.insert(state.end(), leads[p.j].begin(), leads[p.j].end());
  }
}


double LeadMonomialsEnv::step(int action) {
  double reward = env.step(env.P[action]);
  if (leads.size() < env.G.size())
    leads.push_back(lead_monomials_vector(env.G[env.G.size()-1], k, n));
  state.clear();
  for (const auto& p : env.P) {
    state.insert(state.end(), leads[p.i].begin(), leads[p.i].end());
    state.insert(state.end(), leads[p.j].begin(), leads[p.j].end());
  }  
  return reward;
}
