#include "buchberger.h"
#include "ideals.h"

#include <iostream>

int main() {
  std::vector<Polynomial> F = cyclic(7);

  auto G = buchberger(F);
  G = minimalize(G);
  G = interreduce(G);

  std::cout << "\n" << G.size() << "\n";
}
