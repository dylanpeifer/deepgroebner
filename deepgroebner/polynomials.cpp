#include "polynomials.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <tuple>


Coefficient operator/(Coefficient c1, Coefficient c2) {

  // compute a = c2 inverse using extended Euclidean algorithm
  int a = 0, a_ = 1;
  int b = P, b_ = c2.c;
  while (b_ != 0) {
    int q = b / b_;
    std::tie(a, a_) = std::make_tuple(a_, a - q * a_);
    std::tie(b, b_) = std::make_tuple(b_, b - q * b_);
  }

  return c1.c * a;
}

Monomial::Monomial(std::initializer_list<int> exp) {
  if (exp.size() != N) throw std::invalid_argument("exponent vector is wrong size");
  std::copy(exp.begin(), exp.end(), exponent.begin());
  degree = std::accumulate(exponent.begin(), exponent.end(), 0);
}

Monomial operator*(const Monomial& m1, const Monomial& m2) {
  Monomial m;
  m.degree = m1.degree + m2.degree;
  for (int i = 0; i < N; i++)
    m[i] = m1[i] + m2[i];
  return m;
}

Monomial operator/(const Monomial& m1, const Monomial& m2) {
  Monomial m;
  m.degree = m1.degree - m2.degree;
  for (int i = 0; i < N; i++) {
    m[i] = m1[i] - m2[i];
  }
  return m;
}

bool operator>(const Monomial& m1, const Monomial& m2) {
  if (m1.degree > m2.degree) {
    return true;
  } else if (m2.degree > m1.degree) {
    return false;
  } else {
    for (int i = N-1; i >= 0; i--) {
      if (m2.exponent[i] > m1.exponent[i])
	return true;
      else if (m1.exponent[i] > m2.exponent[i])
	return false;
    }
  }
  return false; 
}

bool operator==(const Monomial& m1, const Monomial& m2) {
  for (int i = 0; i < N; i++)
    if (m1.exponent[i] != m2.exponent[i]) return false;
  return true;
}

std::ostream& operator<<(std::ostream& os, const Monomial& m) {
  os << "x^[";
  for (int i = 0; i < N-1; i++)
    os << m[i] << " ";
  os << m[N-1] << "]";
  return os;
}

bool is_divisible(const Monomial& m1, const Monomial& m2) {
  for (int i = 0; i < N; i++) {
    if (m1[i] < m2[i]) return false;
  }
  return true;
}

Monomial gcd(const Monomial& m1, const Monomial& m2) {
  Monomial m;
  for (int i = 0; i < N; i++) {
    m[i] = std::min(m1[i], m2[i]);
    m.degree += m[i];
  }
  return m;
}

Monomial lcm(const Monomial& m1, const Monomial& m2) {
  Monomial m;
  for (int i = 0; i < N; i++) {
    m[i] = std::max(m1[i], m2[i]);
    m.degree += m[i];
  }
  return m;
}

std::ostream& operator<<(std::ostream& os, const Term& t) {
  if (t.monom == Monomial{}) {
    os << t.coeff;
    return os;
  }
  os << t.coeff << t.monom;
  return os;
}

Polynomial::Polynomial(std::initializer_list<Term> tms) {
  for (Term t : tms)
    terms.push_back(t);
  std::sort(terms.begin(), terms.end(),
	    [](const Term& t1, const Term& t2) { return t1.monom > t2.monom; });
}

Polynomial operator+(const Polynomial& f1, const Polynomial& f2) {
  Polynomial g;
  int i = 0, j = 0;
  while (i < f1.terms.size() && j < f2.terms.size()) {
    Term t1 = f1.terms[i];
    Term t2 = f2.terms[j];
    if (t1.monom > t2.monom) {
      g.terms.push_back(t1);
      i++;
    } else if (t2.monom > t1.monom) {
      g.terms.push_back(t2);
      j++;
    } else {
      Coefficient c = t1.coeff + t2.coeff;
      if (c != 0)
	g.terms.push_back({c, t1.monom});
      i++;
      j++;
    }
  }
  if (i < f1.terms.size()) {
    for (int k = i; k < f1.terms.size(); k++)
      g.terms.push_back(f1.terms[k]);
  } else {
    for (int k = j; k < f2.terms.size(); k++)
      g.terms.push_back(f2.terms[k]);
  }
  return g;
}

Polynomial operator-(const Polynomial& f1, const Polynomial& f2) {
  Polynomial f = f2;
  for (Term& t : f.terms)
    t.coeff = (-1) * t.coeff;
  return f1 + f;
}

bool operator==(const Polynomial& f1, const Polynomial& f2) {
  if (f1.terms.size() != f2.terms.size()) return false;
  for (int i = 0; i < f1.terms.size(); i++)
    if (f1.terms[i] != f2.terms[i]) return false;
  return true;
}

Polynomial operator*(const Term& t, const Polynomial& f) {
  Polynomial g;
  for (const Term& ft : f.terms)
    g.terms.push_back(t * ft);
  return g;
}

std::ostream& operator<<(std::ostream& os, const Polynomial& f) {
  int n = f.terms.size();
  if (n == 0) {
    os << "0";
    return os;
  }
  for (int i = 0; i < n-1; i++)
    os << f.terms[i] << " + "; 
  os << f.terms[n-1]; 
  return os;
}
