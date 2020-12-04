#include <vector>

#include "gtest/gtest.h"

#include "polynomials.h"
#include "buchberger.h"


TEST(spoly, example1) {
  Polynomial f = {{1, {1,2,1}},
		  {3, {1,0,1}},
		  {7, {0,0,0}}};
  Polynomial g = {{9, {7,0,0}},
		  {-3, {1,0,1}},
		  {1, {1,0,0}}};
  Polynomial s = {{3, {7,0,1}},
		  {7, {6,0,0}},
		  {10668, {1,2,2}},
		  {28447, {1,2,1}}};
  EXPECT_EQ(spoly(f, g), s);
}


TEST(spoly, example2) {
  Polynomial f = {{1, {2,0}},
		  {1, {1,1}}};
  Polynomial g = {{1, {0,2}},
		  {1, {1,1}}};
  Polynomial s = {};
  EXPECT_EQ(spoly(f, g), s);
}


TEST(spoly, example3) {
  Polynomial f = {{ 1, {3,2}},
		  {-1, {2,3}}};
  Polynomial g = {{ 1, {4,1}},
		  { 1, {0,2}}};
  Polynomial s = {{-1, {3,3}},
		  {-1, {0,3}}};
  EXPECT_EQ(spoly(f, g), s);
}


TEST(spoly, example4) {
  Polynomial f = {{ 1, {2,0}},
		  { 1, {0,3}}};
  Polynomial g = {{ 1, {1,2}},
		  { 1, {1,0}},
		  { 1, {0,0}}};
  Polynomial s = {{ 1, {3,0}},
		  {-1, {1,1}},
		  {-1, {0,1}}};
  EXPECT_EQ(spoly(f, g), s);
}


TEST(reduce, example1) {
  Polynomial g = {{1, {3,1,2}}, {1, {2,0,1}}};
  std::vector<Polynomial> F = {{{1, {2,0,0}}, {1, {0,1,0}}},
                               {{1, {1,1,1}}, {1, {0,0,1}}},
                               {{1, {1,0,2}}, {1, {0,2,0}}}};
  Polynomial r = {{1, {0,1,2}}, {-1, {0,1,1}}};
  EXPECT_EQ(reduce(g, F).first, r);
}


TEST(reduce, example2) {
  Polynomial g = {{1, {5,10,4}}, {22982, {3,1,2}}};
  std::vector<Polynomial> F = {{{1, {5,12,0}}, {25797, {1,5,2}}},
			       {{1, {1, 3,1}}, {27630, {2,1,0}}},
			       {{1, {1, 9,1}}, { 8749, {2,0,0}}}};
  Polynomial r = {{2065, {9,2,0}}, {22982, {3,1,2}}};
  auto [r_, stats] = reduce(g, F);
  EXPECT_EQ(r, r_);
  EXPECT_EQ(stats.steps, 4);
}


TEST(update, example1) {
  Polynomial f = {{1, {2,0}}, {1, {1,1}}, {2, {0,0}}};
  std::vector<Polynomial> F;
  std::vector<SPair> P;

  F.clear();
  P.clear();
  update(F, P, f, EliminationType::None);
  EXPECT_EQ(F, std::vector<Polynomial>({f}));
  EXPECT_TRUE(P.empty());
  
  F.clear();
  P.clear();
  update(F, P, f, EliminationType::LCM);
  EXPECT_EQ(F, std::vector<Polynomial>({f}));
  EXPECT_TRUE(P.empty());

  F.clear();
  P.clear();
  update(F, P, f, EliminationType::GebauerMoeller);
  EXPECT_EQ(F, std::vector<Polynomial>({f}));
  EXPECT_TRUE(P.empty());
}


TEST(minimalize, example1) {
  std::vector<Polynomial> G = {{{ 1, {1,2,0}}, {  1, {0,0,1}}},
			       {{ 1, {1,0,1}}, {  3, {0,1,0}}},
			       {{ 1, {2,0,0}}, {  1, {0,1,1}}},
			       {{-3, {0,3,0}}, {  1, {0,2,0}}},
			       {{-9, {0,1,0}}, { -1, {0,0,3}}},
			       {{ 1, {0,0,8}}, {243, {0,0,1}}}};
  std::vector<Polynomial> Gmin = {{{ 1, {1,0,1}}, { 3, {0,1,0}}},
				  {{ 1, {2,0,0}}, { 1, {0,1,1}}},
				  {{-1, {0,0,3}}, {-9, {0,1,0}}},
				  {{-3, {0,3,0}}, { 1, {0,2,0}}},
				  {{ 1, {1,2,0}}, { 1, {0,0,1}}}};
  EXPECT_EQ(minimalize(G), Gmin);
}


TEST(interreduce, example1) {
  std::vector<Polynomial> G = {{{ 1, {1,0,1}}, { 3, {0,1,0}}},
			       {{ 1, {2,0,0}}, { 1, {0,1,1}}},
			       {{-1, {0,0,3}}, {-9, {0,1,0}}},
			       {{-3, {0,3,0}}, { 1, {0,2,0}}},
			       {{ 1, {1,2,0}}, { 1, {0,0,1}}}};
  std::vector<Polynomial> Gred = {{{1, {1,0,1}}, {    3, {0,1,0}}},
				  {{1, {2,0,0}}, {    1, {0,1,1}}},
				  {{1, {0,0,3}}, {    9, {0,1,0}}},
				  {{1, {0,3,0}}, {21335, {0,2,0}}},
				  {{1, {1,2,0}}, {    1, {0,0,1}}}};
  EXPECT_EQ(interreduce(G), Gred);
}
