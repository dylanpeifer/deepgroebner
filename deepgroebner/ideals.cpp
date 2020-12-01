#include "polynomials.h"
#include "ideals.h"

#include <algorithm>
#include <array>
#include <random>

std::vector<Polynomial> cyclic(int n) {
  if ((n < 1) || (n > N))
    throw std::invalid_argument("n is too large");
  std::vector<Polynomial> F;
  for (int d = 1; d < n; d++) {
    std::vector<Term> p;
    for (int i = 0; i < n; i++) {
      std::array<int, N> exp = {};
      for (int k = 0; k < d; k++) {
	exp[(i+k) % n] = 1;
      }
      p.push_back({1, exp});
    }
    F.push_back(p);
  }
  std::array<int, N> exp = {};
  for (int i = 0; i < n; i++) {
    exp[i] = 1;
  }
  F.push_back({{1, exp}, {-1, {}}});
  return F;
}

std::vector<Monomial> basis(int n, int d) {
  if ((n < 1) || (n > N))
    throw std::invalid_argument("n is too large");

  // arr of bars and stars
  std::vector<int> arr;
  for (int i = 0; i < d; i++)
    arr.push_back(0);
  for (int i = 0; i < n - 1; i++)
    arr.push_back(1);

  std::vector<Monomial> B;
  do {
    std::array<int, N> exp = {};
    int index = 0;
    for (int i = 0; i < arr.size(); i++) {
      if (arr[i] == 0)
	exp[index]++;
      else
	index++;
    }
    B.push_back(exp);
  } while (std::next_permutation(arr.begin(), arr.end()));

  return B;
}

RandomBinomialIdealGenerator::RandomBinomialIdealGenerator(int n, int d, int s,
							   DegreeDistribution D,
							   bool constants,
							   bool homogeneous,
							   bool pure)
  : s(s), pure(pure), homogeneous(homogeneous)
{
  for (int i = 0; i < d+1; i++)
    bases.push_back(basis(n, i));

  std::vector<int> count;
  for (auto& basis : bases) {
    switch (D) {
    case DegreeDistribution::Uniform:
      count.push_back(basis.size());
      break;
    case DegreeDistribution::Weighted:
      count.push_back(1);
      break;
    case DegreeDistribution::Maximum:
      count.push_back(0);
      break;
    }
  }
  if (!constants)
    count[0] = 0;
  if (D == DegreeDistribution::Maximum)
    count[count.size()-1] = 1;
  degree_dist = {count.begin(), count.end()};

  std::random_device rand;
  rng.seed(rand());
}

std::vector<Polynomial> RandomBinomialIdealGenerator::next() {
  std::vector<Polynomial> F;

  std::vector<Monomial> samples;
  for (int i = 0; i < s; i++) {
    Coefficient c = (pure) ? -1 : coeff_dist(rng);
    int d1, d2;
    if (homogeneous) {
      d1 = d2 = degree_dist(rng);
    } else {
      d1 = degree_dist(rng);
      d2 = degree_dist(rng);
    }

    bool success = false;
    for (int trials = 0; trials < 1000; trials++) {
      std::sample(bases[d1].begin(), bases[d1].end(),
		  std::back_inserter(samples), 1, rng);
      Monomial m1 = samples.back();
      samples.pop_back();
      std::sample(bases[d2].begin(), bases[d2].end(),
		  std::back_inserter(samples), 1, rng);
      Monomial m2 = samples.back();
      samples.pop_back();
      if (m1 < m2) {
	F.push_back({{1, m2}, {c, m1}});
	success = true;
	break;
      } else if (m1 > m2) {
	F.push_back({{1, m1}, {c, m2}});
	success = true;
	break;
      }
    }
    if (!success)
      throw std::runtime_error("failed to generate two distinct random monomials after 1000 trials");
  }
  return F;
}
