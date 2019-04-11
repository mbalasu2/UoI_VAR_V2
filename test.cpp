#include <iostream>
#include <eigen3/Eigen/Dense>
#include <assert.h>

using namespace std;
using namespace Eigen;

float explained_variable_R2(VectorXf y_hat, VectorXf y_obs)
{

  //assert(y_hat.size()==y_obs.size());
  float y_dash = y_obs.mean();
  float SSE_tot = (y_obs.array() - y_dash).square().sum();
  float SSE_res = (y_obs - y_hat).array().square().sum();
  //float SSE_reg = (y_hat-y_obs).array().square().sum();
  //float R2 = SSE_reg/SSE_tot;
  float R2 = 1 - (SSE_res/SSE_tot);
  
  return R2;
}


int main()
{

VectorXf X(5);
  X << 15, 18, 21, 24, 27; 

VectorXf Y(5);
Y << 25, 25, 27, 31, 32;

float r2 = explained_variable_R2(X,Y);

cout << "r2: " << r2 << endl; 

/*MatrixXf m(3,3);
m << 0, 1, 2,
              3, 4, 5,
              6, 7, 8;

cout << "V.iszero()? :" <<  V.isZero() << endl;
cout << "m.iszero()? :" << m.isZero() << endl;
cout << "m.isZero(1)? :" << m.isZero(1) << endl;
cout << "m.row(0).isZero? :" << m.row(0).isZero() << endl;
cout << "m " << m << endl;
cout << "m.trans " << m.transpose() << endl;*/



}
