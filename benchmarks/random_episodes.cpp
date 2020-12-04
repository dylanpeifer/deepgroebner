#include <iostream>
#include <random>

#include "buchberger.h"


int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: random_episodes <dist> <episodes>" << std::endl;
    return 1;
  }

  LeadMonomialsEnv env{argv[1], false, true, 2};
  std::default_random_engine rng;

  double total_reward = 0.0;
  int episodes = std::stoi(argv[2]);

  for (int i = 0; i < episodes; i++) {
    env.reset();
    while (!env.state.empty()) {
      std::uniform_int_distribution<int> dist{0, env.state.size() / env.cols - 1};
      int action = dist(rng);
      double reward = env.step(action);
      total_reward += reward;
    }
  }

  std::cout << total_reward / episodes << std::endl;
}
