#ifndef VAR_VECTORIZE_H
#define VAR_VECTORIZE_H

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#ifdef __cplusplus
        extern "C" {
#endif
	//VectorXf var_vectorize (float* d, int local, int yrows, int n_rows, int n_cols, int D,  int L, MPI_Comm comm_r, MPI_Comm comm_world, MPI_Comm comm_group); 
  VectorXf var_vectorize (float* d, int local, int yrows, int n_rows, int n_cols, int D, MPI_Comm comm_r,  MPI_Comm comm_world, MPI_Comm comm_group, int n_readers);



#ifdef __cplusplus
 }
#endif


#endif
