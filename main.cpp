#include <mlpack/core.hpp>

#include "cocbo.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: cocbo K\n";
    return 1;
  }

  size_t k = strtoul(argv[1], nullptr, 10);

  arma::mat data(2, 30, arma::fill::none);
  for (arma::uword i = 0; i < 10; i++)
    data.col(i) = arma::randn(2) + arma::vec({1, 0});
  for (arma::uword i = 10; i < 20; i++)
    data.col(i) = arma::randn(2) + arma::vec({-1, -1});
  for (arma::uword i = 20; i < 30; i++)
    data.col(i) = arma::randn(2) + arma::vec({-1, 1});

  arma::Row<size_t> assignments;
  arma::mat centroids;
  ClusterWithCocbo(data, k, k, k + 1, assignments, centroids);

  std::cout << "centroids = [";
  for (arma::uword c = 0; c < centroids.n_cols; c++) {
    std::cout << "[";
    for (arma::uword r = 0; r < centroids.n_rows; r++)
      std::cout << centroids(r, c) << ",";
    std::cout << "],";
  }
  std::cout << "]\n";

  std::cout << "clusters = [";
  for (size_t c = 0; c < centroids.n_cols; c++) {
    std::cout << "[";
    for (arma::uword i = 0; i < assignments.size(); i++) {
      if (assignments(i) == c) {
        std::cout << "[";
        for (arma::uword r = 0; r < data.n_rows; r++)
          std::cout << data(r, i) << ",";
        std::cout << "],";
      }
    }
    std::cout << "],";
  }
  std::cout << "]\n";

  return 0;
}
