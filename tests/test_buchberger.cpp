#include "polynomials.h"
#include "buchberger.h"

#include "gtest/gtest.h"

TEST(spoly, example1) {
  Polynomial f = {{1, {1,2,1,0,0,0,0,0}},
		  {3, {1,0,1,0,0,0,0,0}},
		  {7, {0,0,0,0,0,0,0,0}}};
  Polynomial g = {{9, {7,0,0,0,0,0,0,0}},
		  {-3, {1,0,1,0,0,0,0,0}},
		  {1, {1,0,0,0,0,0,0,0}}};
  Polynomial s = {{3, {7,0,1,0,0,0,0,0}},
		  {7, {6,0,0,0,0,0,0,0}},
		  {10668, {1,2,2,0,0,0,0,0}},
		  {28447, {1,2,1,0,0,0,0,0}}};
  EXPECT_EQ(spoly(f, g), s);
}

TEST(reduce, example1) {
  Polynomial g = {{1, {3,1,2,0,0,0,0,0}},
		  {1, {2,0,1,0,0,0,0,0}}};
  Polynomial f1 = {{1, {2,0,0,0,0,0,0,0}},
		   {1, {0,1,0,0,0,0,0,0}}};
  Polynomial f2 = {{1, {1,1,1,0,0,0,0,0}},
		   {1, {0,0,1,0,0,0,0,0}}};
  Polynomial f3 = {{1, {1,0,2,0,0,0,0,0}},
		   {1, {0,2,0,0,0,0,0,0}}};
  std::vector<Polynomial> F = {f1, f2, f3};
  Polynomial r = {{1, {0,1,2,0,0,0,0,0}},
		  {-1, {0,1,1,0,0,0,0,0}}};
  EXPECT_EQ(reduce(g, F), r);
}
