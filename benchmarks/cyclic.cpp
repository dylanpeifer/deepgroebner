#include <iostream>

#include "buchberger.h"
#include "ideals.h"


int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: cyclic <number>" << std::endl;
    return 1;
  }
  std::vector<Polynomial> F = cyclic(std::stoi(argv[1]));
  std::vector<Polynomial> G = buchberger(F);
  std::cout << G.size() << std::endl;
}
