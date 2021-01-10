#include <iostream>

#include "buchberger.h"
#include "ideals.h"


int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: cyclic <number>" << std::endl;
    return 1;
  }
  auto F = cyclic(std::stoi(argv[1]));
  auto result = buchberger(F);
  std::cout << result.first.size() << std::endl;
}
