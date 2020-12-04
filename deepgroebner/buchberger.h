/**
 * An environment for computing Groebner bases with Buchberger's algorithm.
 */

#ifndef BUCHBERGER_H
#define BUCHBERGER_H

#include <map>
#include <memory>
#include <vector>

#include "ideals.h"
#include "polynomials.h"


/**
 * Return the s-polynomial of f and g.
 */
Polynomial spoly(const Polynomial& f, const Polynomial& g);


/**
 * Stores statistics computed during a call to reduce. For now, just an int counting steps.
 */
struct ReduceStats {
  int steps;
};


/**
 * Return a remainder and stats when g is divided by polynomials F.
 *
 * @param g Dividend polynomial.
 * @param F Divisor polynomials.
 */
std::pair<Polynomial, ReduceStats> reduce(const Polynomial& g, const std::vector<Polynomial>& F);


/**
 * Stores information about an s-pair. For now, just the indices.
 */
struct SPair {
  int i;
  int j;
};


inline bool operator==(const SPair& p1, const SPair& p2) { return (p1.i == p2.i) && (p1.j == p2.j); }


/**
 * Elimination option for managing the pair set.
 *
 * Strategy can be None (eliminate no pairs), LCM (only eliminate pairs that
 * fail the LCM criterion), or GebauerMoeller (use full Gebauer-Moeller elimination).
 */
enum class EliminationType {GebauerMoeller, LCM, None};


/**
 * Update the polynomials and pairs when f is added to the basis G.
 *
 * @param G Current polynomial generators.
 * @param P Current s-pairs.
 * @param f New polynomial to add to the basis.
 * @param elimination Strategy for pair elimination.
 */
void update(std::vector<Polynomial>& G, std::vector<SPair>& P, const Polynomial& f, EliminationType elimination);


/**
 * Return a minimal Groebner basis from arbitrary Groebner basis G.
 */
std::vector<Polynomial> minimalize(const std::vector<Polynomial>& G);


/**
 * Return the reduced Groebner basis from minimal Groebner basis G.
 */
std::vector<Polynomial> interreduce(const std::vector<Polynomial>& G);


/**
 * Return the Groebner basis for the ideal generated by F using Buchberger's algorithm.
 */
std::vector<Polynomial> buchberger(const std::vector<Polynomial>& F, EliminationType elimination = EliminationType::GebauerMoeller);


/**
 * Reward option for the environment.
 *
 * Option can be Additions to count each polynomial addition (including one for generating the s-polynomial)
 * or Reductions to count number of s-pair reductions.
 */
enum class RewardType {Additions, Reductions};


/**
 * An environment for computing a Groebner basis using Buchberger's algorithm.
 */
class BuchbergerEnv {

public:

  /**
   * @param ideal_dist String naming the ideal distribution.
   * @param elimination Strategy for pair elimination.
   * @param rewards Reward value for each step.
   * @param sort_input Whether to sort the initial generating set by lead monomial.
   * @param sort_reducers Whether to choose reducers in sorted order by lead monomial.
   */
  BuchbergerEnv(std::string ideal_dist = "3-20-10-uniform",
		EliminationType elimination = EliminationType::GebauerMoeller,
		RewardType rewards = RewardType::Additions,
		bool sort_input = false,
		bool sort_reducers = true);

  // define destructor/copy/move since std::unique_ptr has no copy constructor
  ~BuchbergerEnv() = default;
  BuchbergerEnv(const BuchbergerEnv& other) : ideal_gen(other.ideal_gen->copy()) {}
  BuchbergerEnv& operator=(const BuchbergerEnv& other) { ideal_gen = other.ideal_gen->copy(); return *this; }
  BuchbergerEnv(BuchbergerEnv&& other) = default;
  BuchbergerEnv& operator=(BuchbergerEnv&& other) = default;

  void reset();

  double step(SPair action);

  void seed(int seed) { ideal_gen->seed(seed); }

  int nvars() const { return ideal_gen->nvars(); }

  std::vector<Polynomial> G;
  std::vector<SPair> P;

private:

  std::unique_ptr<IdealGenerator> ideal_gen;
  EliminationType elimination;
  RewardType rewards;
  bool sort_input;
  bool sort_reducers;

};


/**
 * Return the concatenated exponent vectors of the k lead monomials of f.
 *
 * @param f 
 * @param n Number of variables.
 * @param k Number of lead terms to concatenate.
 */
std::vector<int> lead_monomials_vector(const Polynomial& f, int n, int k);


/**
 * A BuchbergerEnv with state the matrix of pairs' lead monomials.
 */
class LeadMonomialsEnv {

public:

  /**
   * @param ideal_dist String naming the ideal distribution.
   * @param sort_input Whether to sort the initial generating set by lead monomial.
   * @param sort_reducers Whether to choose reducers in sorted order by lead monomial.
   * @param k The number of lead monomials shown for each polynomial.
   */
  LeadMonomialsEnv(std::string ideal_dist = "3-20-10-uniform",
		   bool sort_input = false,
		   bool sort_reducers = true,
		   int k = 2);

  void reset();

  double step(int action);

  void seed(int seed) { env.seed(seed); };

  std::vector<int> state;
  int cols;

private:

  BuchbergerEnv env;
  int n;
  int k;
  std::vector<std::vector<int>> leads;

};

#endif
