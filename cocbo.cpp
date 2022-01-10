/*
* Unofficial COCBO algorithm implementation
* Copyright (C) 2022 azyobuzin
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <fmt/core.h>
#include <glpk.h>

#include <cassert>
#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/sample_initialization.hpp>
#include <stdexcept>

struct ScopedGlpProb {
  glp_prob *lp;

  ~ScopedGlpProb() {
    glp_delete_prob(lp);
    lp = nullptr;
  }
};

void ClusterWithCocbo(const arma::mat &data, size_t k, size_t lower_bound,
                      size_t upper_bound, arma::Row<size_t> &assignments,
                      arma::mat &centroids, size_t max_iterations = 1000) {
  if (data.empty()) throw std::invalid_argument("data is empty");
  if (k == 0 || lower_bound > k || upper_bound <= k)
    throw std::invalid_argument("k is out of range");

  // 最適化問題を解ける条件を満たす K であることを確認
  size_t n_cluster = data.n_cols / k;
  if (data.n_cols > (k + 1) * n_cluster)
    throw std::invalid_argument("can't assign cluster with the specified k");

  glp_prob *lp = glp_create_prob();
  ScopedGlpProb _sgp{lp};  // 自動で glp_delete_prob する

  glp_set_prob_name(lp, "COCBO");
  glp_set_obj_dir(lp, GLP_MIN);  // 最小化問題

  // 帰属度 u_ki
  glp_add_cols(lp, data.n_cols * n_cluster);
  for (size_t k = 0; k < data.n_cols; k++) {
    for (size_t i = 0; i < n_cluster; i++) {
      std::string col_name = fmt::format("u_{},{}", k, i);
      glp_set_col_name(lp, k * n_cluster + i + 1, col_name.c_str());
      glp_set_col_kind(lp, k * n_cluster + i + 1, GLP_BV);  // 0 or 1
    }
  }

  glp_add_rows(lp, data.n_cols * n_cluster);

  {
    std::vector<int> indices(data.n_cols + 1);
    std::vector<double> ones(data.n_cols + 1, 1);

    // 制約1: 各要素（k）はいずれかのクラスタに属している
    for (size_t k = 0; k < data.n_cols; k++) {
      int row_idx = k + 1;
      std::string row_name = fmt::format("sum(u_{},i)=1", k);
      glp_set_row_name(lp, row_idx, row_name.c_str());
      glp_set_row_bnds(lp, row_idx, GLP_FX, 1, 1);
      for (size_t i = 0; i < n_cluster; i++)
        indices[i + 1] = k * n_cluster + i + 1;
      glp_set_mat_row(lp, row_idx, n_cluster, indices.data(), ones.data());
    }

    // 制約2: 各クラスタには [lower_bound, upper_bound] 個の要素が属している
    for (size_t i = 0; i < n_cluster; i++) {
      int row_idx = data.n_cols + i + 1;
      std::string row_name =
          fmt::format("{} <= sum(u_k,{}) <= {}", lower_bound, i, upper_bound);
      glp_set_row_name(lp, row_idx, row_name.c_str());
      glp_set_row_bnds(lp, row_idx, GLP_DB, lower_bound, upper_bound);
      for (size_t k = 0; k < data.n_cols; k++)
        indices[k + 1] = k * n_cluster + i + 1;
      glp_set_mat_row(lp, row_idx, data.n_cols, indices.data(), ones.data());
    }
  }

  // 初期重心を選択
  mlpack::kmeans::SampleInitialization().Cluster(data, n_cluster, centroids);

  arma::mat new_centroids(centroids.n_rows, centroids.n_cols, arma::fill::none);
  std::vector<size_t> assign_count(n_cluster);
  mlpack::metric::EuclideanDistance metric;

  assignments.zeros(data.n_cols);

  for (size_t i = 0; i < max_iterations; i++) {
    // 目的関数を設定
    for (size_t k = 0; k < data.n_cols; k++) {
      for (size_t i = 0; i < n_cluster; i++) {
        double distance = metric.Evaluate(data.col(k), centroids.col(i));
        glp_set_obj_coef(lp, k * n_cluster + i + 1, distance);
      }
    }

    // 最適化
    int solve_result = glp_simplex(lp, nullptr);
    if (solve_result != 0) {
      throw std::runtime_error(
          fmt::format("glp_simplex returned {}", solve_result));
    }

    // 結果取得
    for (size_t k = 0; k < data.n_cols; k++) {
      for (size_t i = 0;; i++) {
        double u = glp_get_col_prim(lp, k * n_cluster + i + 1);
        assert(u == 0 || u == 1);
        if (u == 1) {
          assignments[k] = i;
          break;
        }
        if (i >= n_cluster - 1) {
          throw std::runtime_error(
              fmt::format("{}th object is not assigned to any cluster", k));
        }
      }
    }

    // クラスタ重心の更新
    new_centroids.zeros();
    std::fill(assign_count.begin(), assign_count.end(), 0);
    for (size_t k = 0; k < data.n_cols; k++) {
      size_t i = assignments(k);
      assign_count[i]++;
      new_centroids.col(i) += data.col(k);
    }
    for (size_t i = 0; i < n_cluster; i++) {
      new_centroids.col(i) /= assign_count[i];
    }
    if (arma::approx_equal(new_centroids, centroids, "absdiff", 1e-5)) {
      // クラスタ重心が更新されなければ終わり
      break;
    }
    std::swap(centroids, new_centroids);
  }
}
