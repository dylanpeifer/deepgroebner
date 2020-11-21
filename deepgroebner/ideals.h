#ifndef IDEALS_H
#define IDEALS_H

#include "polynomials.h"

#include <random>
#include <vector>

std::vector<Polynomial> cyclic(int n);

class FixedIdealGenerator {
public:
  FixedIdealGenerator(std::vector<Polynomial> P) : F{P} {}
  std::vector<Polynomial> next() { return F; }
private:
  std::vector<Polynomial> F;
};

std::vector<Monomial> basis(int n, int i);

enum class DegreeDistribution {Uniform, Weighted, Maximum};

class RandomBinomialIdealGenerator {
public:
  RandomBinomialIdealGenerator(int n, int d, int s,
			       DegreeDistribution D = DegreeDistribution::Uniform,
			       bool constants = false,
			       bool homogeneous = false,
			       bool pure = false);
  std::vector<Polynomial> next();
  void seed(int seed) { rng.seed(seed); }
private:
  int s;
  bool pure;
  bool homogeneous;
  std::vector<std::vector<Monomial>> bases;
  std::default_random_engine rng;
  std::discrete_distribution<int> degree_dist;
  std::uniform_int_distribution<int> coeff_dist {1, P-1};
};

#endif
