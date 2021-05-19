/**
 * Generators of ideals.
 */

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <random>
#include <sstream>

#include "ideals.h"
#include "polynomials.h"


std::vector<Polynomial> cyclic(int n) {
  if (n > N) throw std::invalid_argument("n is too large");
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
  if (n > N) throw std::invalid_argument("n is too large");

  // use vector of bars (1) and stars (0) to correspond to monomial in degree d
  std::vector<int> a;
  for (int i = 0; i < d; i++)
    a.push_back(0);
  for (int i = 0; i < n - 1; i++)
    a.push_back(1);

  // all permutations of vector correspond to all monomials in degree d
  std::vector<Monomial> B;
  do {
    std::array<int, N> exp = {};
    int index = 0;
    for (int i = 0; i < a.size(); i++) {
      if (a[i] == 0)
	exp[index]++;
      else
	index++;
    }
    B.push_back(exp);
  } while (std::next_permutation(a.begin(), a.end()));

  return B;
}


int binomial(int n, int k) {
  if ((k == 0) || (k == n))
    return 1;
  else
    return binomial(n - 1, k - 1) + binomial(n - 1, k);
}


std::discrete_distribution<int> degree_distribution(int n, int d, DistributionType dist, bool constants) {
  std::vector<int> count;

  if (constants)
    count.push_back(1);
  else
    count.push_back(0);

  switch (dist) {
  case DistributionType::Uniform:
    for (int i = 1; i < d + 1; i++)
      count.push_back(binomial(n + i - 1, n - 1));
    break;
  case DistributionType::Weighted:
    for (int i = 0; i < d; i++)
      count.push_back(1);
    break;
  case DistributionType::Maximum:
    for (int i = 0; i < d - 1; i++)
      count.push_back(0);
    count.push_back(1);
    break;
  }

  return {count.begin(), count.end()};
}


std::unique_ptr<IdealGenerator> parse_ideal_dist(const std::string& ideal_dist) {

  // split ideal_dist on '-'
  std::vector<std::string> dist_args;
  std::string arg;
  std::istringstream iss(ideal_dist);
  while (std::getline(iss, arg, '-'))
    dist_args.push_back(arg);

  std::map<std::string, DistributionType> dist_types = {
      {"uniform", DistributionType::Uniform},
      {"weighted", DistributionType::Weighted},
      {"maximum", DistributionType::Maximum}
  };

  if (dist_args[0] == "cyclic") {
    int n = std::stoi(dist_args[1]);
    return std::make_unique<FixedIdealGenerator>(cyclic(n));

  } else if (dist_types.find(dist_args[3]) != dist_types.end()) {
    int n = std::stoi(dist_args[0]);
    int d = std::stoi(dist_args[1]);
    int s = std::stoi(dist_args[2]);
    DistributionType dist = dist_types[dist_args[3]];
    bool constants = (std::find(dist_args.begin(), dist_args.end(), "consts") != dist_args.end());
    bool homogeneous = (std::find(dist_args.begin(), dist_args.end(), "homog") != dist_args.end());
    bool pure = (std::find(dist_args.begin(), dist_args.end(), "pure") != dist_args.end());
    return std::make_unique<RandomBinomialIdealGenerator>(n, d, s, dist, constants, homogeneous, pure);

  } else {
    int n = std::stoi(dist_args[0]);
    int d = std::stoi(dist_args[1]);
    int s = std::stoi(dist_args[2]);
    double lam = std::stod(dist_args[3]);
    DistributionType dist = dist_types[dist_args[4]];
    bool constants = (std::find(dist_args.begin(), dist_args.end(), "consts") != dist_args.end());
    bool homogeneous = (std::find(dist_args.begin(), dist_args.end(), "homog") != dist_args.end());
    return std::make_unique<RandomIdealGenerator>(n, d, s, lam, dist, constants, homogeneous);

  }
}


FixedIdealGenerator::FixedIdealGenerator(std::vector<Polynomial> F) : F(F) {
  n = 0;
  for (const Polynomial& f : F) {
    for (const Term& t : f.terms) {
      for (int i = 0; i < N; i++)
	if (t.monom[i] != 0) n = std::max(n, i);
    }
  }
}


RandomBinomialIdealGenerator::RandomBinomialIdealGenerator(int n, int d, int s, DistributionType dist,
							   bool constants, bool homogeneous, bool pure)
    : n(n), s(s), homogeneous(homogeneous), pure(pure) {
  for (int i = 0; i < d + 1; i++)
    bases.push_back(basis(n, i));
  degree_dist = degree_distribution(n, d, dist, constants);
  std::random_device rand;
  rng.seed(rand());
}


std::vector<Polynomial> RandomBinomialIdealGenerator::next() {
  std::vector<Polynomial> F;
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
      Monomial m1 = *choice(bases[d1].begin(), bases[d1].end(), rng);
      Monomial m2 = *choice(bases[d2].begin(), bases[d2].end(), rng);
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


RandomIdealGenerator::RandomIdealGenerator(int n, int d, int s, double lam, DistributionType dist, bool constants, bool homogeneous)
    : n(n), s(s), length_dist(lam), homogeneous(homogeneous) {
  for (int i = 0; i < d + 1; i++)
    bases.push_back(basis(n, i));
  degree_dist = degree_distribution(n, d, dist, constants);
  std::random_device rand;
  rng.seed(rand());
}


std::vector<Polynomial> RandomIdealGenerator::next() {
  std::vector<Polynomial> F;
  for (int i = 0; i < s; i++) {
    Polynomial f;
    int terms = 2 + length_dist(rng);
    int d = degree_dist(rng);
    for (int j = 0; j < terms; j++) {
      Coefficient c = coeff_dist(rng);
      Monomial m = *choice(bases[d].begin(), bases[d].end(), rng);
      f = f + Polynomial{{c, m}};
      if (!homogeneous)
	d = degree_dist(rng);
    }
    Term t = {1 / f.LC(), {}};
    F.push_back(t * f);
  }
  return F;
}
