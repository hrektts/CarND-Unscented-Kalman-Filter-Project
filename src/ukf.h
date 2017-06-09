#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;


  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

private:
  ///* augmented sigma points matrix
  MatrixXd Xsig_aug_;

  ///* measurement dimension
  int n_z_;

  ///* sigma points matrix in measurement space
  MatrixXd Zsig_;

  ///* predicted mean measurement
  VectorXd z_pred_;

  ///* measurement covariance matrix
  MatrixXd S_;

  /**
   * Compute augmented sigma points
   */
  void ComputeAugmentedSigmaPoints();

  /**
   * Predict augmented sigma points
   * @param delta_t Time difference between current and previous predictions
   */
  void PredictAugmentedSigmaPoints(double delta_t);

  /**
   * Predict mean and covariance based on process model
   */
  void PredictMeanAndCovariance();

  /**
   * Transform predicted state into the measurement space
   * @param Zsig Sigma points in measurement space
   * @param z_pred Predicted mean measurement
   * @param S Measurement covariance matrix
   */
  void PredictLaserMeasurement(MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S);

  /**
   * Update the belief about the object's position using laser data
   * @param z Measured radar data
   * @param Zsig Sigma points in measurement space
   * @param z_pred Predicted mean measurement
   * @param S Measurement covariance matrix
   */
  void UpdateLaserState(const VectorXd& z, const MatrixXd& Zsig,
                        const VectorXd& z_pred, const MatrixXd& S);

  /**
   * Transform predicted state into the measurement space of the lidar
   * @param R Measurement covariance noise matrix
   * @param angles The indexes of which the unit of the measurement is an angle
   */
  void PredictLidarMeasurement(const MatrixXd& R, std::vector<unsigned> angles);

  /**
   * Transform predicted state into the measurement space of the radar
   * @param R Measurement covariance noise matrix
   * @param angles The indexes of which the unit of the measurement is an angle
   */
  void PredictRadarMeasurement(const MatrixXd& R, std::vector<unsigned> angles);

  /**
   * Transform predicted state into the measurement space
   * @param R Measurement covariance noise matrix
   * @param angles The indexes of which the unit of the measurement is an angle
   */
  void PredictMeasurementCommon(const MatrixXd& R, std::vector<unsigned> angles);

  /**
   * Update the belief about the object's position using radar data
   * @param z Measured radar data
   * @param x_angles The indexes of which the unit of the state is an angle
   * @param z_angles The indexes of which the unit of the measurement is an angle
   */
  void UpdateState(const VectorXd& z, std::vector<unsigned> x_angles, std::vector<unsigned> z_angles);
};

#endif /* UKF_H */
