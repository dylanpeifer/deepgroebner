#include "buchberger.h"

#include <iostream>

int main() {
  BuchbergerEnv env {3, 20, 10, DistributionType::Uniform};
  float total_reward = 0.0;
  int episodes = 10000;

  for (int i = 0; i < episodes; i++) {
    env.reset();
    while (!env.P.empty()) {
      SPair action = env.P[0];
      float reward = env.step(action);
      total_reward += reward;
    }
  }

  std::cout << total_reward / episodes << std::endl;
}
