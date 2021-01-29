/* Generate CSV file of strategy performance for sampled ideals. */

#include <iostream>
#include <fstream>
#include <sstream>

#include "buchberger.h"
#include "ideals.h"
#include "polynomials.h"


std::vector<Polynomial> parse_ideal_string(const std::string& ideal_string) {
  std::vector<Polynomial> F;
  std::istringstream iss(ideal_string);
  std::string poly_string;
  while (std::getline(iss, poly_string, '|'))
    F.push_back(parse_polynomial(poly_string));
  return F;
}


int main(int argc, char* argv[]) {

  if (argc < 3) {
    std::cout << "Usage: make_strat <distribution> <strategy> <seed>" << std::endl;
    return 1;
  }
  std::string dist = argv[1];
  std::string strat = argv[2];
  int seed = (argc > 3) ? std::stoi(argv[3]) : 0;

  std::ifstream in_file {"data/stats/" + dist + "/" + dist + ".csv"};
  if(!in_file.good()) {
    std::cout << "No distribution file found. Run scripts/make_dist.m2 first." << std::endl;
    return 2;
  }

  std::string out_filename = "data/stats/" + dist + "/" + dist + "_" + strat + ".csv";
  if (std::ifstream{out_filename}) {
    std::cout << "Output file " << out_filename
	      << " already exists. Delete or move it first." << std::endl;
    return 3;
  }
  std::ofstream out_file {out_filename};
  out_file << "ZeroReductions,NonzeroReductions,PolynomialAdditions" << std::endl;

  std::map<std::string, SelectionType> select_map = {
      {"first", SelectionType::First},
      {"degree", SelectionType::Degree},
      {"normal", SelectionType::Normal},
      {"sugar", SelectionType::Sugar},
  };
  SelectionType select = select_map[strat];

  std::string line;
  std::getline(in_file, line);  // remove column name
  while(std::getline(in_file, line)) {
    auto F = parse_ideal_string(line);
    auto [G, stats] = buchberger(F, select);
    out_file << stats.zero_reductions << ","
             << stats.nonzero_reductions << ","
             << stats.polynomial_additions << std::endl;
  }
}
