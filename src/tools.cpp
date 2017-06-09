#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd::Zero(4);

  if (estimations.size() != ground_truth.size()
      || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  int l = estimations.size();
  for (int i = 0; i < l; i++) {
    VectorXd residual = estimations[i] - ground_truth[i];

    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse = (rmse / l).array().sqrt();

  return rmse;
}

void Tools::RoundRadian(VectorXd& vec, vector<unsigned> index) {
  for (auto idx : index) {
    while (vec(idx) > M_PI) {
      vec(idx) -= 2. * M_PI;
    }
    while (vec(idx) < -M_PI) {
      vec(idx) += 2. * M_PI;
    }
  }
}
