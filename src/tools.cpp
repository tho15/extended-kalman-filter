#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	/**
	* Calculate the RMSE here.
	*/
	VectorXd rmse(4);
	rmse << 0,0,0,0;
 
	if ( estimations.size() != ground_truth.size()
	     || estimations.size() == 0 ) {
	     std::cout << "Invalid estimation or ground_truth data" << std::endl;
	     return rmse;
	}


	//accumulate squared residuals
	for (unsigned int i=0; i < estimations.size(); ++i) {
		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();
	
	return rmse; 
}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
	MatrixXd Hj(3,4);
	//recover state parameters
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);

	double p2   = px*px + py*py;
	double p12  = sqrt(p2);
	double p32  = p2*p12;
	//check division by zero
	if (std::fabs(p2) < 0.00001) {
	    std::cout<<"CalculateJacobian error: divided by 0!"<< std::endl;
	    Hj << 0, 0, 0, 0,
	          0, 0, 0, 0,
	          0, 0, 0, 0;
	    return Hj;
	}
	//compute the Jacobian matrix
	Hj << px/p12, py/p12, 0, 0,
	      -py/p2, px/p2,  0, 0,
	      py*(vx*py-vy*px)/p32, px*(vy*px-vx*py)/p32, px/p12, py/p12;

	return Hj;  
}
