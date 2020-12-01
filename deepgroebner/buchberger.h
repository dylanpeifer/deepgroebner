#ifndef BUCHBERGER_H
#define BUCHBERGER_H

#include "polynomials.h"
#include "ideals.h"

#include <map>
#include <queue>
#include <vector>


Polynomial spoly(const Polynomial& f, const Polynomial& g);
std::pair<Polynomial, int> reduce(const Polynomial& g, const std::vector<Polynomial>& F);

struct SPair {
  int i;
  int j;
};

enum class EliminationStrategy {GebauerMoeller, LCM, None};

void update(std::vector<Polynomial>& G, std::vector<SPair>& P, const Polynomial& f, EliminationStrategy elimination);

std::vector<Polynomial> minimalize(std::vector<Polynomial>& G);
std::vector<Polynomial> interreduce(std::vector<Polynomial>& G);
std::vector<Polynomial> buchberger(const std::vector<Polynomial>& F, EliminationStrategy elimination = EliminationStrategy::GebauerMoeller);

class SPairSet {
public:
  SPairSet(std::vector<Polynomial>& gens) : F{gens} {}
  int size() { return sz; }
  bool empty() const { return sz == 0; }
  SPair pop();
  void update(const Polynomial& r);

private:
  int sz = 0;
  std::vector<Polynomial>& F;
  std::priority_queue<std::tuple<int, int, int>> keys;
  std::map<std::tuple<int, int, int>, std::vector<SPair>> bins;
};

enum class RewardOption {Additions, Reductions};

class BuchbergerEnv {
public:
  BuchbergerEnv(int n, int d, int s,
		DegreeDistribution D = DegreeDistribution::Uniform,
		bool constants = false,
		bool homogeneous = false,
		bool pure = false,
		EliminationStrategy elimination = EliminationStrategy::GebauerMoeller,
		RewardOption rewards = RewardOption::Additions,
		bool sort_input = false,
		bool sort_reducers = true);
  void reset();
  float step(SPair action);
  void seed(int seed) { ideal_gen.seed(seed); }

  std::vector<Polynomial> F;
  std::vector<SPair> P;
private:
  RandomBinomialIdealGenerator ideal_gen;
  EliminationStrategy elimination;
  RewardOption rewards;
  bool sort_input;
  bool sort_reducers;
  std::vector<Polynomial> reducers;
};

class LeadMonomialsEnv {
public:
  LeadMonomialsEnv() {}
  void reset() {}
  float step(int action) { return 2.0; }
  void seed(int seed) {}
  int state[12];
};

#endif
