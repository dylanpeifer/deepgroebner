#include "polynomials.h"

#include "gtest/gtest.h"

TEST(Coefficient, constructor) {
  Coefficient x = 2045;
  Coefficient y = -2;
  Coefficient z = 32008;
  EXPECT_EQ(x, 2045);
  EXPECT_EQ(y, 32001);
  EXPECT_EQ(z, 5);
}

TEST(Coefficient, add) {
  Coefficient x = 3;
  Coefficient y = 10;
  Coefficient z = 13;
  EXPECT_EQ(x + y, z);
}

TEST(Coefficient, subtract) {
  Coefficient x = 3;
  Coefficient y = 10;
  Coefficient z = 7;
  EXPECT_EQ(y - x, z);
}

TEST(Coefficient, multiply) {
  Coefficient x = 3;
  Coefficient y = 10;
  Coefficient z = -2;
  EXPECT_EQ(x * y, 30);
  EXPECT_EQ(x * z, 31997);
}

TEST(Coefficient, divide) {
  Coefficient x1 = 3;
  Coefficient y1 = 10;
  EXPECT_EQ(x1 / y1, 28803);
  Coefficient x2 = 23002;
  Coefficient y2 = 32001;
  EXPECT_EQ(x2 / y2, 20502);
  Coefficient x3 = 12000;
  Coefficient y3 = 4;
  EXPECT_EQ(x3 / y3, 3000);
  Coefficient x4 = 12345;
  Coefficient y4 = 1;
  EXPECT_EQ(x4 / y4, 12345);
}

TEST(Monomial, constructor) {
  Monomial m1;
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(m1[i], 0);
  }
  Monomial m2 = {1,2,3,4,5,6,7,8};
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(m2[i], i+1);
  }
}

TEST(Monomial, multiply) {
  Monomial m1 = {1,2,3,4,5,6,7,8};
  Monomial m2 = {2,3,4,5,6,7,8,9};
  Monomial m3 = {3,5,7,9,11,13,15,17};
  EXPECT_EQ(m1 * m2, m3);
}

TEST(Monomial, divide) {
  Monomial m1 = {2,3,4,5,6,7,8,9};
  Monomial m2 = {1,2,3,4,5,6,7,8};
  Monomial m3 = {1,1,1,1,1,1,1,1};
  EXPECT_EQ(m1 / m2, m3);
}

TEST(Monomial, order) {
  Monomial m1 = {1,1,1,1,1,1,1,1};
  Monomial m2 = {0,0,0,0,0,0,0,9};
  Monomial m3 = {0,0,0,0,2,2,2,2};
  EXPECT_LT(m3, m1);
  EXPECT_LT(m1, m2);
  EXPECT_LT(m3, m2);
  EXPECT_GT(m1, m3);
  EXPECT_GT(m2, m1);
  EXPECT_GT(m2, m3);
}

TEST(Monomial, is_divisible) {
  Monomial m1 = {1,2,3,4,5,6,7,8};
  Monomial m2 = {0,1,2,3,4,5,6,7};
  Monomial m3 = {1,1,1,1,1,1,1,1};
  EXPECT_TRUE(is_divisible(m1, m1));
  EXPECT_TRUE(is_divisible(m1, m2));
  EXPECT_TRUE(is_divisible(m1, m3));
  EXPECT_FALSE(is_divisible(m2, m1));
  EXPECT_TRUE(is_divisible(m2, m2));
  EXPECT_FALSE(is_divisible(m2, m3));
  EXPECT_FALSE(is_divisible(m3, m1));
  EXPECT_FALSE(is_divisible(m3, m2));
  EXPECT_TRUE(is_divisible(m3, m3));
}

TEST(Monomial, gcd) {
  Monomial m1 = {1,1,1,1,0,0,0,0};
  Monomial m2 = {0,0,0,0,1,1,1,1};
  Monomial m3 = {0,0,0,0,0,0,0,0};
  Monomial m4 = {2,2,2,2,2,2,2,2};
  EXPECT_EQ(gcd(m1, m2), m3);
  EXPECT_EQ(gcd(m2, m1), m3);
  EXPECT_EQ(gcd(m1, m4), m1);
  EXPECT_EQ(gcd(m2, m4), m2);
}

TEST(Monomial, lcm) {
  Monomial m1 = {2,2,2,2,0,0,0,0};
  Monomial m2 = {0,0,0,0,2,2,2,2};
  Monomial m3 = {2,2,2,2,2,2,2,2};
  Monomial m4 = {1,1,1,1,1,1,1,1};
  EXPECT_EQ(lcm(m1, m2), m3);
  EXPECT_EQ(lcm(m2, m1), m3);
  EXPECT_EQ(lcm(m3, m4), m3);
}

TEST(Term, constructor) {
  Term t = {3, {1,1,1,1,1,1,1,1}};
  Coefficient c = 3;
  Monomial m = {1,1,1,1,1,1,1,1};
  EXPECT_EQ(t.coeff, c);
  EXPECT_EQ(t.monom, m);
}

TEST(Term, multiply) {
  Term t1 = {3, {1,1,1,1,1,1,1,1}};
  Term t2 = {7, {0,1,1,1,1,1,1,1}};
  Term t3 = {21, {1,2,2,2,2,2,2,2}};
  EXPECT_EQ(t1 * t2, t3);
}

TEST(Term, divide) {
  Term t1 = {3, {1,1,1,1,1,1,1,1}};
  Term t2 = {7, {0,1,1,1,1,1,1,1}};
  Term t3 = {13716, {1,0,0,0,0,0,0,0}};
  EXPECT_EQ(t1 / t2, t3);
}

TEST(Polynomial, constructor) {
  Polynomial p = {{1, {1,1,1,1,1,1,1,1}},
                  {3, {0,0,0,0,1,1,1,1}},
		  {9, {1,1,2,2,3,4,1,1}},
                  {1, {0,0,0,0,0,0,0,0}}};
  Monomial m = {1,1,2,2,3,4,1,1};
  EXPECT_EQ(p.LC(), 9);
  EXPECT_EQ(p.LM(), m);
}

TEST(Polynomial, add) {
  Polynomial p1 = {{ 1, {1,2,1}},
		   { 3, {1,0,1}},
		   { 7, {0,0,0}}};
  Polynomial p2 = {{ 9, {7,0,0}},
		   {-3, {1,0,1}},
		   { 1, {1,0,0}}};
  Polynomial p3 = {{ 9, {7,0,0}},
		   { 1, {1,2,1}},
		   { 1, {1,0,0}},
		   { 7, {0,0,0}}};
  EXPECT_EQ(p1 + p2, p3);
}

TEST(Polynomial, subtract) {
  Polynomial p1 = {{ 1, {1,2,1}},
		   { 3, {1,0,1}},
		   { 7, {0,0,0}}};
  Polynomial p2 = {{ 9, {7,0,0}},
		   {-3, {1,0,1}},
		   { 1, {1,0,0}}};
  Polynomial p3 = {{ 9, {7,0,0}},
		   { 1, {1,2,1}},
		   { 1, {1,0,0}},
		   { 7, {0,0,0}}};
  EXPECT_EQ(p3 - p2, p1);
  EXPECT_EQ(p3 - p1, p2);
  EXPECT_EQ(p1 - p1, Polynomial{});
}

TEST(Polynomial, multiply_term) {
  Monomial m = {1,1,2,2,5,6,1,1};
  Term t1 = {9, {0,0,0,0,2,2,0,0}};
  Term t2 = {2, {0,0,0,0,0,0,0,0}};
  Polynomial p1 = {{1, {1,1,1,1,1,1,1,1}},
                   {3, {0,0,0,0,1,1,1,1}},
		   {9, {1,1,2,2,3,4,1,1}},
                   {1, {0,0,0,0,0,0,0,0}}};
  Polynomial p2 = {{9,  {1,1,1,1,3,3,1,1}},
                   {27, {0,0,0,0,3,3,1,1}},
		   {81, {1,1,2,2,5,6,1,1}},
                   {9,  {0,0,0,0,2,2,0,0}}};
  EXPECT_EQ((t1 * p1).LC(), 81);
  EXPECT_EQ((t1 * p1).LM(), m);
  EXPECT_EQ(t1 * p1, p2);
  EXPECT_EQ(t2 * p1, p1 + p1);
  EXPECT_EQ(t2 * p2, p2 + p2);
}

TEST(Polynomial, multiply_polynomial) {
  Polynomial p1 = {{1, {1,2,0}},
		   {1, {0,1,1}},
		   {1, {0,0,0}}};
  Polynomial p2 = {{1, {1,1,1}},
		   {1, {1,0,0}}};
  Polynomial p3 = {{1, {2,3,1}},
		   {1, {1,2,2}},
		   {1, {2,2,0}},
		   {2, {1,1,1}},
		   {1, {1,0,0}}};
  EXPECT_EQ(p1 * p2, p3);
}
