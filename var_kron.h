#ifndef VAR_KRON_H
#define VAR_KRON_H

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#ifdef __cplusplus
        extern "C" {
#endif
	MatrixXf var_kron (float* d, int local, int yrows, int n_rows, int n_cols, int D,  MPI_Comm comm_readers, MPI_Comm comm_world, MPI_Comm comm_group, int n_readers); 



#ifdef __cplusplus
 }
#endif


#endif
