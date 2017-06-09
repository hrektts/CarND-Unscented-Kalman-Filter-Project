#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.7;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  is_initialized_ = false;

  n_x_ = x_.size();

  // augmented state vector: [ (state vector) longitudinal_acc_noise yaw_acc_noise ]
  n_aug_ = n_x_ + 2;

  lambda_ = 3 - n_aug_;

  P_ = MatrixXd::Identity(n_x_, n_x_);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  weights_ = VectorXd(2 * n_aug_ + 1);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    x_ = VectorXd::Ones(5);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = meas_package.raw_measurements_(0);
      x_(0) = meas_package.raw_measurements_(1);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double ro = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      x_(0) = ro * cos(phi);
      x_(1) = ro * sin(phi);
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  double dt_s = (meas_package.timestamp_ - time_us_) / 1000000.;
  time_us_ = meas_package.timestamp_;

  Prediction(dt_s);

  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  ComputeAugmentedSigmaPoints();
  PredictAugmentedSigmaPoints(delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  n_z_ = 2;

  std::vector<unsigned> x_angles {3};
  std::vector<unsigned> z_angles {};

  MatrixXd R = MatrixXd::Zero(n_z_, n_z_);
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;

  PredictLidarMeasurement(R, z_angles);

  VectorXd z = VectorXd(2);
  z = meas_package.raw_measurements_.head(2);
  UpdateState(z, x_angles, z_angles);

  VectorXd z_diff = z - z_pred_;
  // NIS output
  //std::cout<< z_diff.transpose() * S_.inverse() * z_diff << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  n_z_ = 3;

  std::vector<unsigned> x_angles {3};
  std::vector<unsigned> z_angles {1};

  MatrixXd R = MatrixXd::Zero(n_z_, n_z_);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;

  PredictRadarMeasurement(R, z_angles);

  VectorXd z = VectorXd(3);
  z = meas_package.raw_measurements_.head(3);
  UpdateState(z, x_angles, z_angles);

  VectorXd z_diff = z - z_pred_;
  // NIS output
  //std::cout<< z_diff.transpose() * S_.inverse() * z_diff << std::endl;
}

/**
 * Compute augmented sigma points
 */
void UKF::ComputeAugmentedSigmaPoints() {
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(5) = x_;

  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  MatrixXd L_P_aug = P_aug.llt().matrixL();

  Xsig_aug_.col(0) = x_aug;
  Xsig_aug_.block(0, 1, n_aug_, n_aug_)
      = (sqrt(lambda_ + n_aug_) * L_P_aug).colwise() + x_aug;
  Xsig_aug_.block(0, (n_aug_ + 1), n_aug_, n_aug_)
      = MatrixXd::Zero(n_aug_, n_aug_).colwise()
      + x_aug - sqrt(lambda_ + n_aug_) * L_P_aug;
}

/**
 * Predict augmented sigma points
 * @param delta_t Time difference between current and previous predictions
 */
void UKF::PredictAugmentedSigmaPoints(double delta_t) {
  int l = Xsig_aug_.cols();
  for (int i = 0; i < l; i++) {
    VectorXd x = Xsig_aug_.col(i);

    double pos_0, pos_1;
    if (fabs(x(4)) < 0.0001) {
      pos_0 = x(0) + x(2) * cos(x(3)) * delta_t + delta_t * delta_t * cos(x(3)) * x(5) / 2.;
      pos_1 = x(1) + x(2) * -sin(x(3)) * delta_t + delta_t * delta_t * sin(x(3)) * x(5) / 2.;
    } else {
      pos_0 = x(0) + x(2) / x(4) * (sin(x(3) + x(4) * delta_t) - sin(x(3)))
           + delta_t * delta_t * cos(x(3)) * x(5) / 2.;
      pos_1 = x(1) + x(2) / x(4) * (-cos(x(3) + x(4) * delta_t) + cos(x(3)))
           + delta_t * delta_t * sin(x(3)) * x(5) / 2.;
    }
    double vel_abs = x(2) + delta_t * x(5);
    double yaw_angle = x(3) + x(4) * delta_t + delta_t * delta_t * x(6) / 2;
    double yaw_rate = x(4) + delta_t * x(6);

    Xsig_pred_.col(i) << pos_0, pos_1, vel_abs, yaw_angle, yaw_rate;
  }
}

/**
 * Predict mean and covariance based on process model
 */
void UKF::PredictMeanAndCovariance() {
  //set weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < weights_.size(); i++) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  //predict state mean
  x_.fill(0.);
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  P_.fill(0.);
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
    VectorXd diff = Xsig_pred_.col(i) - x_;
    while (diff(3) > M_PI) {
      diff(3) -= 2. * M_PI;
    }
    while (diff(3) < -M_PI) {
      diff(3) += 2. * M_PI;
    }

    P_ += weights_(i) * diff * diff.transpose();
  }
}

/**
 * Transform predicted state into the measurement space of the lidar
 * @param R Measurement covariance noise matrix
 * @param angles The indexes of which the unit of the measurement is an angle
 */
void UKF::PredictLidarMeasurement(const MatrixXd& R, std::vector<unsigned> angles) {
  //create matrix for sigma points in measurement space
  Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);
  Zsig_ = Xsig_pred_.topLeftCorner(n_z_, Zsig_.cols());

  PredictMeasurementCommon(R, angles);
}

/**
 * Transform predicted state into the measurement space of the radar
 * @param R Measurement covariance noise matrix
 * @param angles The indexes of which the unit of the measurement is an angle
 */
void UKF::PredictRadarMeasurement(const MatrixXd& R, std::vector<unsigned> angles) {
  //create matrix for sigma points in measurement space
  Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
    VectorXd x = Xsig_pred_.col(i);
    double rho = sqrt(x(0) * x(0) + x(1) * x(1));
    double phi = atan2(x(1), x(0));
    while (phi > M_PI) {
      phi -= 2. * M_PI;
    }
    while (phi < -M_PI) {
      phi += 2. * M_PI;
    }
    double rho_dot = (x(0) * cos(x(3)) * x(2) + x(1) * sin(x(3)) * x(2)) / rho;
    Zsig_.col(i) << rho, phi, rho_dot;
  }

  PredictMeasurementCommon(R, angles);
}

/**
 * Transform predicted state into the measurement space
 * @param R Measurement covariance noise matrix
 * @param angles The indexes of which the unit of the measurement is an angle
 */
void UKF::PredictMeasurementCommon(const MatrixXd& R, std::vector<unsigned> angles) {
  //mean predicted measurement
  z_pred_ = VectorXd::Zero(n_z_);
  for (int i = 0; i < Zsig_.cols(); i++) {
    z_pred_ += weights_(i) * Zsig_.col(i);
  }

  //measurement covariance matrix
  S_ = MatrixXd::Zero(n_z_, n_z_);
  Tools t;
  for (int i = 0; i < Zsig_.cols(); i++) {
    VectorXd diff = Zsig_.col(i) - z_pred_;
    t.RoundRadian(diff, angles);

    S_ += weights_(i) * diff * diff.transpose();
  }

  if (R.rows() != n_z_ && R.cols() != n_z_) {
    cout << "Invalid dimension of measurement covariance noise matrix" << endl;
    return;
  }

  S_ += R;
}

/**
 * Update the belief about the object's position using radar data
 * @param z Measured radar data
 * @param x_angles The indexes of which the unit of the state is an angle
 * @param z_angles The indexes of which the unit of the measurement is an angle
 */
void UKF::UpdateState(const VectorXd& z,
                      std::vector<unsigned> x_angles, std::vector<unsigned> z_angles) {
  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_);
  Tools t;
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    t.RoundRadian(x_diff, x_angles);

    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    t.RoundRadian(z_diff, z_angles);

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S_.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred_;
  t.RoundRadian(z_diff, z_angles);

  x_ += K * z_diff;
  P_ -= K * S_ * K.transpose();
}
