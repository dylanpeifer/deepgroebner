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

  return {r, {steps}};
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
		      Monomial gam = lcm(G[p.i].LM(), G[p.j].LM());
		      return (is_divisible(gam, f.LM()) &&
			      (gam != lcm(G[p.i].LM(), f.LM())) &&
			      (gam != lcm(G[p.j].LM(), f.LM())));
	            };
    P.erase(std::remove_if(P.begin(), P.end(), can_drop), P.end());

    std::map<Monomial, std::vector<int>> lcms;
    for (int i = 0; i < m; i++) {
      lcms[lcm(G[i].LM(), f.LM())].push_back(i);
    }
    std::vector<Monomial> min_lcms;
    for (const auto& [gam, v] : lcms) {  // will be in sorted order because std::map
      if (std::all_of(min_lcms.begin(), min_lcms.end(), [&gam](const Monomial& m) { return !is_divisible(gam, m); })) {
	min_lcms.push_back(gam);
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


std::vector<Polynomial> buchberger(const std::vector<Polynomial>& F, EliminationType elimination) {

  std::vector<Polynomial> G;
  std::vector<SPair> P;
  for (const Polynomial& f : F) {
    update(G, P, f, elimination);
  }

  while (!P.empty()) {
    auto iter = std::min_element(P.begin(), P.end(),
				 [&G](const SPair& a, const SPair& b) {
				     return lcm(G[a.i].LM(), G[a.j].LM()) < lcm(G[b.i].LM(), G[b.j].LM());
				 });
    SPair p = *iter;
    P.erase(iter);
    auto [r, stats] = reduce(spoly(G[p.i], G[p.j]), G);
    if (r.size() != 0) {
      update(G, P, r, elimination);
    }
  }

  return interreduce(minimalize(G));
}


BuchbergerEnv::BuchbergerEnv(std::string ideal_dist,
			     EliminationType elimination,
			     RewardType rewards,
			     bool sort_input,
			     bool sort_reducers)
    : elimination(elimination), rewards(rewards), sort_input(sort_input), sort_reducers(sort_reducers) {
  ideal_gen = parse_ideal_dist(ideal_dist);
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
  if (rewards == RewardType::Additions)
    return -1.0 * (stats.steps + 1);
  else
    return -1.0;
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
