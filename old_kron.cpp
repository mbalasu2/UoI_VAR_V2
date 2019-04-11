#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <iostream>
#include <fstream>
#include <mpi.h>
#define EIGEN_USE_MKL_ALL
#include <eigen3/Eigen/Dense>
#include <unistd.h>
#include <string.h>
#include "bins.h"
#include "var_kron.h"

using namespace Eigen; 
using namespace std; 

void print_array1 (int *vec, int rows, char name[]) {

  int leni;
  FILE *fp;
  fp = fopen(name, "w");

  for (leni =0; leni < rows; leni++) {
    fprintf(fp, "%d\n", *(vec + leni));
  }

  fclose (fp);

}

void print1( MatrixXf m, string name )
{
  std::ofstream file(name);
  if (file.is_open())
  {
    file  << m << '\n';
  }

}

MatrixXf var_kron (float *d, int local, int yrows, int n_rows, int n_cols, int D,  MPI_Comm comm_world, MPI_Comm comm_group)
{
  int i, j;
  size_t sized = (size_t) local * (size_t) (n_cols) * sizeof(float);

  int rank_world, nprocs_world;
  MPI_Comm_rank(comm_world, &rank_world);
  MPI_Comm_size(comm_world, &nprocs_world);

  int size_group, rank_group;
  MPI_Comm_size(comm_group, &size_group);
  MPI_Comm_rank(comm_group, &rank_group);

  //int sized = d.rows() * d.cols() * sizeof(float);

  MPI_Win win;
  MPI_Win_create(d, sized, sizeof(float), MPI_INFO_NULL, comm_world, &win); 

  int *sample;
  if (rank_group==0) {
    sample = (int *)malloc((n_rows-D)* n_cols * sizeof(int)); 
    for (int i=0; i<(n_rows-D)*n_cols;i++) sample[i] = i; 

    print_array1(sample, (n_rows-D)*n_cols, "./debug/sample_kron.dat");
  } else {
    sample = NULL;
  }

  int srows[yrows];

  {
    int sendcounts[size_group];
    int displs[size_group];

    for (i=0; i<size_group; i++) {
      int ubound;
      bin_range_1D(i, (n_rows-D) * n_cols, size_group, &displs[i], &ubound); 
      sendcounts[i] = bin_size_1D(i, (n_rows-D), size_group) * n_cols; 
    }

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &srows, yrows, MPI_INT, 0, comm_group);

    if(rank_group==0) free(sample);
  }



  double t = MPI_Wtime();
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

  //MatrixXf B_out(yrows, D*n_cols*n_cols);
  //B_out.setZero();  

  float *B_out;
  B_out = (float*) malloc(yrows * D * n_cols * n_cols * sizeof(float) );

  //VectorXf y(n_cols);
  //float *y;
  //y = (float *)malloc(n_cols * sizeof(float));

  for (i=0; i<yrows; i++) {
#ifdef SIMPLESAMPLE
    int trow = (int) random_at_mostL( (long) n_rows);
#else
    int trow = srows[i] % (n_rows-D);
#endif
    int target_rank = bin_coord_1D(trow, n_rows-D, nprocs_world);
    int target_disp = bin_index_1D(trow, n_rows-D, nprocs_world) * n_cols;
    int col_disp = srows[i] / (n_rows-D);

    MPI_Get(&B_out[n_cols * (i + col_disp)], n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);
  }

  //print1(B_out, "./debug/B_out.txt"); 


  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

  MPI_Win_free(&win);

  /*float *out;
    out = (float *)malloc(B_out.rows() * B_out.cols() * sizeof(float));
    Map<Matrix<float, Dynamic, Dynamic, RowMajor> >(out, B_out.rows(), B_out.cols()) = B_out; */

  Map<MatrixXf> out (B_out, yrows, D * n_cols * n_cols); 
  return out;
}
