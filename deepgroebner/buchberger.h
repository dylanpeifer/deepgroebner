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
		DistributionType D = DistributionType::Uniform,
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

std::vector<int> lead_monomials_vector(const Polynomial& f, int k, int n);

class LeadMonomialsEnv {
public:
  LeadMonomialsEnv(int n = 3, int d = 20, int s = 10,
		   DistributionType D = DistributionType::Uniform,
		   bool constants = false,
		   bool homogeneous = false,
		   bool pure = false,
		   EliminationStrategy elimination = EliminationStrategy::GebauerMoeller,
		   RewardOption rewards = RewardOption::Additions,
		   bool sort_input = false,
		   bool sort_reducers = true,
		   int k = 2);
  void reset();
  float step(int action);
  void seed(int seed) { env.seed(seed); };

  std::vector<int> state;
private:
  BuchbergerEnv env;
  int k;
  int n;
  int m;
  std::vector<std::vector<int>> leads;
};

#endif
