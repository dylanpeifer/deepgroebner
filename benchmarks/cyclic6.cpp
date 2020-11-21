#include "buchberger.h"
#include "ideals.h"

#include <iostream>

int main() {
  std::vector<Polynomial> F = cyclic(6);

  for (auto& f : F) std::cout << f << std::endl;

  auto G = buchberger(F);
  G = minimalize(G);
  G = interreduce(G);

  std::cout << "\n" << G.size() << "\n\n";
  for (auto& g : G) std::cout << g << "\n";
  std::cout << std::endl;
}
