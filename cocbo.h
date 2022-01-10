#include <mlpack/core.hpp>

void ClusterWithCocbo(const arma::mat &data, size_t k, size_t lower_bound,
                      size_t upper_bound, arma::Row<size_t> &assignments,
                      arma::mat &centroids, size_t max_iterations = 1000);
