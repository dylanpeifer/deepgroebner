#include "buchberger.h"
#include "ideals.h"
#include "polynomials.h"

#include <algorithm>
#include <map>
#include <queue>
#include <vector>


Polynomial spoly(const Polynomial& f, const Polynomial& g) {
  Term gamma = {1, lcm(f.LM(), g.LM())};
  return (gamma / f.LT()) * f - (gamma / g.LT()) * g;
}

Polynomial reduce(const Polynomial& g, const std::vector<Polynomial>& F) {
  Polynomial r;
  Polynomial h = g;

  while (h.size() != 0) {
    bool found_divisor = false;

    for (const Polynomial& f : F) {
      if (is_divisible(h.LM(), f.LM())) {
	h = h - (h.LT() / f.LT()) * f;
	found_divisor = true;
	break;
      }
    }

    if (!found_divisor) {
      r = r + Polynomial{h.LT()};
      h = h - Polynomial{h.LT()};
    }

  }

  return r;
}

void update(std::vector<Polynomial>& G, std::vector<SPair>& P, const Polynomial& f, EliminationStrategy elimination) {

  std::vector<SPair> P_;
  
  switch (elimination) {
  case EliminationStrategy::None:
    for (int i = 0; i < G.size(); i++) {
      P_.push_back(SPair{i, G.size()});
    }
    break;
  case EliminationStrategy::LCM:
    for (int i = 0; i < G.size(); i++) {
      if (lcm(G[i].LM(), f.LM()) != G[i].LM() * f.LM())
	P_.push_back(SPair{i, G.size()});
    }
    break;
  case EliminationStrategy::GebauerMoeller:
    auto fn = [&G, &f](const SPair& p) {
		Monomial gam = lcm(G[p.i].LM(), G[p.j].LM());
		return is_divisible(gam, f.LM()) && (gam != lcm(G[p.i].LM(), f.LM())) && (gam != lcm(G[p.j].LM(), f.LM()));
	      };
    P.erase(std::remove_if(P.begin(), P.end(), fn), P.end());

    std::map<Monomial, std::vector<int>> lcms;
    for (int i = 0; i < G.size(); i++) {
      lcms[lcm(G[i].LM(), f.LM())].push_back(i);
    }
    std::vector<Monomial> min_lcms;
    for (auto& p : lcms) {
      if (std::all_of(min_lcms.begin(), min_lcms.end(), [&p](const Monomial& m) { return !is_divisible(p.first, m); })) {
	min_lcms.push_back(p.first);
	if (std::all_of(p.second.begin(), p.second.end(), [&G, &f](int i) { return lcm(G[i].LM(), f.LM()) != G[i].LM() * f.LM(); }))
	  P_.push_back(SPair{p.second[0], G.size()});
      }
    }
    break;
  }

  G.push_back(f);
  P.insert(P.end(), P_.begin(), P_.end());
}

std::vector<Polynomial> minimalize(std::vector<Polynomial>& G) {
  std::sort(G.begin(), G.end(), [](const Polynomial& f, const Polynomial& g) { return f.LM() < g.LM(); });
  std::vector<Polynomial> Gmin;
  for (Polynomial& g : G) {
    if (std::none_of(Gmin.begin(), Gmin.end(), [&g](const Polynomial& f) { return is_divisible(g.LM(), f.LM()); })) {
      Gmin.push_back(g);
    }
  }
  return Gmin;
}

std::vector<Polynomial> interreduce(std::vector<Polynomial>& G) {
  for (int i = 0; i < G.size(); i++) {
    Term t = {1 / G[i].LC(), {0,0,0,0,0,0,0,0}};
    G[i] = t * (reduce(G[i] - Polynomial{G[i].LT()}, G) + Polynomial{G[i].LT()});
  }
  return G;
}

std::vector<Polynomial> buchberger(const std::vector<Polynomial>& F, EliminationStrategy elimination) {

  std::vector<Polynomial> G;
  std::vector<SPair> P;

  for (const Polynomial& f : F) {
    update(G, P, f, elimination);
  }

  while (!P.empty()) {
    auto iter = std::min_element(P.begin(), P.end(), [&G](const SPair& a, const SPair& b) {
						       return lcm(G[a.i].LM(), G[a.j].LM()) < lcm(G[b.i].LM(), G[b.j].LM()); });
    SPair p = *iter;
    P.erase(iter);
    Polynomial r = reduce(spoly(G[p.i], G[p.j]), G);
    if (r.size() != 0) {
      update(G, P, r, elimination);
    }
  }

  return G;
}

SPair SPairSet::pop() {
  while (bins[keys.top()].empty()) {
    bins.erase(keys.top());
    keys.pop();
  }
  auto& bin = bins[keys.top()];
  auto p = bin.back();
  bin.pop_back();
  sz--;
  return p;
}

void SPairSet::update(const Polynomial& r) {
  int n = F.size();
  for (int i = 0; i < n; i++) {
    auto key = std::make_tuple(-lcm(F[i].LM(), r.LM()).deg(), -n, -i);
    if (bins.find(key) == bins.end()) {
      keys.push(key);
    }
    bins[key].push_back(SPair{i, n});
    sz++;
  }
}

BuchbergerEnv::BuchbergerEnv(int n, int d, int s, DegreeDistribution D, bool constants, bool homogeneous, bool pure,
			     EliminationStrategy elimination, RewardOption rewards, bool sort_input, bool sort_reducers)
  : ideal_gen(n, d, s, D, constants, homogeneous, pure),
    elimination(elimination), rewards(rewards), sort_input(sort_input), sort_reducers(sort_reducers)
{
}

void BuchbergerEnv::reset() {
  std::vector<Polynomial> I = ideal_gen.next();
  for (auto& f : I) {
    update(F, P, f, elimination);
  }
}

float BuchbergerEnv::step(std::pair<int, int> action) {
  return 2.0;
}
