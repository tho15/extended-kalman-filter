#include <cmath>
#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
  x_ = F_*x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd y  = z - H_*x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S  = H_*P_*Ht + R_;
  MatrixXd K  = P_*Ht*S.inverse();
  MatrixXd I  = MatrixXd::Identity(x_.size(), x_.size());
  
  /* measurement update */
  x_ = x_ + K*y;
  P_ = (I - K*H_)*P_;
}

/**
  * EKF update method need inputs of Jocobian Matrix and
  * measurement noise, the default R is for laser measurement
  * we use Radar measurement for EKF
*/
void KalmanFilter::UpdateEKF(const VectorXd &z, const MatrixXd &Hj, const MatrixXd &Rr) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  double ro, phi, ro_dot;
  double px = x_(0);
  double py = x_(1);
  
  double p2 = px*px + py*py;
  if(std::fabs(p2) < 0.00001) {
  	std::cout << "EKF: position vector magnitude is 0, skip update!" << std::endl;
  	return;
  }
  
  ro  = std::sqrt(p2);
  phi = std::atan2(py, px);
  ro_dot = (x_(0)*x_(2)+x_(1)*x_(3))/std::sqrt(p2);
    
  VectorXd hx(3);
  hx << ro, phi, ro_dot;
  
  VectorXd y = z - hx;
  MatrixXd Ht = Hj.transpose();
  MatrixXd S  = Hj*P_*Ht + Rr;
  MatrixXd K  = P_*Ht*S.inverse();
  MatrixXd I  = MatrixXd::Identity(x_.size(), x_.size());

  /* measurement update */
  x_ = x_ + K*y;
  P_ = (I - K*Hj)*P_;
  
}


