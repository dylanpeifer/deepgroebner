#include "polynomials.h"
#include "buchberger.h"

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

std::vector<Polynomial> buchberger(const std::vector<Polynomial>& F) {

  std::vector<Polynomial> G;
  SPairSet P {G};

  for (Polynomial f : F) {
    P.update(f);
    G.push_back(f);
  }
  
  while (!P.empty()) {
    SPair p = P.pop();
    Polynomial r = reduce(spoly(G[p.i], G[p.j]), G);
    if (r.size() != 0) {
      P.update(r);
      G.push_back(r);
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
