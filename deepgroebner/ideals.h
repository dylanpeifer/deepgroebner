/**
 * Generators of ideals.
 */

#ifndef IDEALS_H
#define IDEALS_H

#include <memory>
#include <random>
#include <vector>

#include "polynomials.h"


/**
 * Return the cyclic-n ideal.
 */
std::vector<Polynomial> cyclic(int n);


/**
 * Return the monomial basis of the polynomial ring in degree d.
 *
 * A monomial basis in degree d is just all monomials of degree d in the
 * polynomial ring.
 *
 * @param n Number of variables in the polynomial ring.
 * @param d Degree of the basis.
 */
std::vector<Monomial> basis(int n, int d);


/**
 * Return the binomial coefficient n choose k.
 */
int binomial(int n, int k);


/**
 * Distribution options on monomials.
 *
 * In a Uniform distribution, monomials are sampled uniformly at random from all monomials less than or
 * equal to the maximum degree (which means higher degrees are more likely). In Weighted, monomials are sampled 
 * with weights such that each degree has the same probability of appearing (i.e., degrees are sampled uniformly
 * and then a monomial is picked uniformly at random from that degree). In Maximum, monomials are sampled uniformly
 * at random from all monomials of the maximum degree.
 */
enum class DistributionType {Uniform, Weighted, Maximum};


/**
 * Return the probability distribution on degrees.
 *
 * @param n Number of variables in the polynomial ring.
 * @param d Maximum degree included in the distribution.
 * @param dist Type of distribution on monomials.
 * @param constants Whether to have nonzero probability of constants.
 */
std::discrete_distribution<int> degree_distribution(int n,
						    int d,
						    DistributionType dist = DistributionType::Uniform,
						    bool constants = false);


/*
 * Return an iterator to an element picked uniformly at random.
 */
template<class Iter>
Iter choice(Iter begin, Iter end, std::default_random_engine& rng) {
  std::uniform_int_distribution<> dist(0, std::distance(begin, end) - 1);
  std::advance(begin, dist(rng));
  return begin;
}


/**
 * Abstract base class for all ideal generators.
 *
 * Derived classes must implement a next method which returns
 * a new std::vector<Polynomial> representing an ideal. If randomness
 * is used to generate polynomials then override the seed method. To
 * ensure a BuchbergerEnv can be copied we also need to define a copy_deriv
 * method that returns a pointer to a new object of the derived class. This
 * is actually very easy - see examples below.
 */
class IdealGenerator {

public:

  virtual std::vector<Polynomial> next() = 0;

  virtual void seed(int seed) {}

  virtual int nvars() const = 0;

  auto copy() const { return std::unique_ptr<IdealGenerator>(copy_deriv()); }

protected:

  virtual IdealGenerator* copy_deriv() const = 0;

};


/*
 * Return pointer to concrete IdealGenerator instance given by string.
 */
std::unique_ptr<IdealGenerator> parse_ideal_dist(const std::string& ideal_dist);


/**
 * Generator of repeated copies of a fixed ideal.
 */
class FixedIdealGenerator : public IdealGenerator {

public:

  /*
   * @param F Polynomial generators for the ideal.
   */
  FixedIdealGenerator(std::vector<Polynomial> F);

  std::vector<Polynomial> next() override { return F; }

  int nvars() const override { return n; }

protected:

  FixedIdealGenerator* copy_deriv() const override { return new FixedIdealGenerator(*this); }

private:

  int n;
  std::vector<Polynomial> F;

};


/**
 * Generator of random examples of binomial ideals.
 */
class RandomBinomialIdealGenerator : public IdealGenerator {

public:

  /**
   * @param n Number of variables in the polynomial ring.
   * @param d Maximum degree of a monomial.
   * @param s Number of generators of each ideal.
   * @param dist Type of distribution on monomials.
   * @param constants Whether to include constants as monomials.
   * @param homogeneous Whether the binomials are homogeneous.
   * @param pure Whether the binomials are pure.
   */
  RandomBinomialIdealGenerator(int n = 3, int d = 20, int s = 10, DistributionType dist = DistributionType::Uniform,
			       bool constants = false, bool homogeneous = false, bool pure = false);

  std::vector<Polynomial> next() override;

  void seed(int seed) override { rng.seed(seed); }

  int nvars() const override { return n; }

protected:

  RandomBinomialIdealGenerator* copy_deriv() const override { return new RandomBinomialIdealGenerator(*this); }

private:

  int n;
  int s;
  bool homogeneous;
  bool pure;
  std::vector<std::vector<Monomial>> bases;
  std::default_random_engine rng;
  std::discrete_distribution<int> degree_dist;
  std::uniform_int_distribution<int> coeff_dist{1, P-1};

};


/**
 * Generator of random examples of polynomial ideals.
 *
 * The number of terms in each polynomial is two larger than
 * a Poisson random variable sampled for each polynomial. Terms
 * are not checked to be distinct, so could sum or cancel with
 * other terms in producing the final polynomial.
 */
class RandomIdealGenerator : public IdealGenerator {

public:

  /*
   * @param n Number of variables in the polynomial ring.
   * @param d Maximum degree of a monomial.
   * @param s Number of generators of each ideal.
   * @param lam Parameter for the Poisson distribution on lengths.
   * @param dist Type of distribution on monomials.
   * @param constants Whether to include constants as monomials.
   * @param homogeneous Whether the polynomials are homogeneous.
   */
  RandomIdealGenerator(int n = 3, int d = 20, int s = 10, double lam = 0.5,
		       DistributionType dist = DistributionType::Uniform,
		       bool constants = false, bool homogeneous = false);

  std::vector<Polynomial> next() override;

  void seed(int seed) override { rng.seed(seed); }

  int nvars() const override { return n; }

protected:

  RandomIdealGenerator* copy_deriv() const override { return new RandomIdealGenerator(*this); }

private:

  int n;
  int s;
  bool homogeneous;
  std::vector<std::vector<Monomial>> bases;
  std::default_random_engine rng;
  std::discrete_distribution<int> degree_dist;
  std::poisson_distribution<> length_dist;
  std::uniform_int_distribution<int> coeff_dist{1, P-1};

};

#endif
