#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <iostream>
#include <fstream>
#include <mpi.h>
#define EIGEN_USE_MKL_ALL
#include <eigen3/Eigen/Dense>
#include <unistd.h>
#include <string.h>
#include "bins.h"
#include "var_vectorize.h"

using namespace Eigen;
using namespace std;

void print_array2 (int *vec, int rows, char name[]) {

  int leni;
  FILE *fp;
  fp = fopen(name, "w");

  for (leni =0; leni < rows; leni++) {
    fprintf(fp, "%d\n", *(vec + leni));
  }

  fclose (fp);

}

void print4( MatrixXf m, string name )
{
  std::ofstream file(name);
  if (file.is_open())
  {
    file  << m << '\n';
  }

}

VectorXf col_stack(MatrixXf m, int  D) {


  //VectorXf out ((m.rows()-D)*m.cols());

  //MatrixXf m_t = m.topRows(m.rows()-1).transpose();


  MatrixXf m_last = m.topRows(m.rows()-D);
  //cout << "before inplace transpose" << endl;
  m_last.transposeInPlace(); 

  //cout << "after inplace transpose" << endl;
  VectorXf out(m_last.rows() * m_last.cols());
   
   //cout << "after out init" << endl;
 
  for (int i=0; i<m_last.rows(); i++)
    out.segment(i*(m_last.cols()),(m_last.cols()))= m_last.row(i);

  //for(int i=0;i<m.rows();i++) {
  //out.segment(i*m.cols(), m.cols()) = m.row(i);
  // }

  return out;

}

/*  * var_vectorize_response function vectorizes/column stacks all but last D rows of bdata_f'. 
 * getting the transpose and then vectorizing the response is pretty time consuming if done in a 
 distributed fashion.
 * So the root rank gets the bdata_f using gatherv. Then the matrix is transposed and all but the
 last D rows are vectorized by col_stack() function.  
 */




VectorXf var_vectorize (float* d, int local, int kron_rows, int n_rows, int n_cols, int D, MPI_Comm comm_r,  MPI_Comm comm_world, MPI_Comm comm_group, int n_readers)
{
  int i, j;
  size_t sized;

  int rank_world, nprocs_world;
  MPI_Comm_rank(comm_world, &rank_world);
  MPI_Comm_size(comm_world, &nprocs_world);

  int size_group, rank_group;
  MPI_Comm_size(comm_group, &size_group);
  MPI_Comm_rank(comm_group, &rank_group);

  int size_r=-1, rank_r=-1;

  //cout << "inside var_vectorize" << endl;

  if (MPI_COMM_NULL != comm_r) 
  {
    MPI_Comm_size(comm_r, &size_r);
    MPI_Comm_rank(comm_r, &rank_r);

  }

  //if( rank_group == 0 )
  //  cout << "passed comm_r size" << endl;

  MatrixXf Y_mat;
  //if(rank_world==0)
  //  Y_mat.setZero(n_rows, n_cols);

  {
    int *recvcounts = NULL;
    int *displs = NULL; 
    int recvsize;
    int Y_rows = local;

    if(MPI_COMM_NULL != comm_r)
        recvcounts = (int*) malloc( size_r * sizeof(int));
  
    //if(rank_group == 0)
    //cout << "before gather" << endl;
  
    if(MPI_COMM_NULL != comm_r)
      MPI_Gather(&Y_rows, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm_r); 

  
    //if(MPI_COMM_NULL != comm_r)
    //  if (rank_r == 0)
    //    cout << "after gather" << endl;

    //since we are gathering matrices we need to multiply by the cols. 
    if(MPI_COMM_NULL != comm_r) {
      /*for(int i=0; i<size_r; i++) {
        recvcounts[i] *= n_col;
        cout << "reccnts: " << recvcounts[i];
      }*/

      //cout << "after recvcount" << endl;
      displs = (int*) malloc( size_r * sizeof(int) );
      displs[0] = 0;
    
      //if(rank_r == 0)
      //  cout << "after displs 0" << endl;

        for (int i=1; i<size_r; i++) 
          displs[i] = displs[i-1] + recvcounts[i-1];

      //cout << "after displs" << endl;

      for (int i=0; i<size_r; i++)
        recvsize += recvcounts[i];


      //if(rank_r == 0)
      //  cout << "after recvsize: "  << recvsize << endl;

    }

    //cout << "before Y_mat set" << endl;
    if(MPI_COMM_NULL != comm_r) {
      if(rank_r == 0) {
        Y_mat.setZero(recvsize, n_cols);
        //cout << "Passed till here in vectorize" << endl; 
      }
      for(int i=0; i<size_r; i++) {
        recvcounts[i] *= n_cols;
      }

    //if(rank_r == 0)
    //  cout << "Passed recvcnts" << endl; 
    }
    //cout << "after Y_mat." << endl;
  
    if(MPI_COMM_NULL != comm_r)
      MPI_Gatherv(d, local*n_cols, MPI_FLOAT, Y_mat.data(), recvcounts, displs, MPI_FLOAT, 0, comm_r);   
  }

 // if( rank_group == 0 )
 //   cout << "Passed Gatherv" << endl;

  VectorXf Y, B_out;
  if (rank_group == 0)
    Y = col_stack(Y_mat, D);

  //if(rank_group == 0) {
  //  print4(Y_mat, "./debug/Y_mat.txt"); 
 //   print4(Y, "./debug/Y_debug.txt");
  //}

  //cout << "after gatherv" << endl;
  {
    int *sendcounts;
    int *displs1;
    //int kron_rows = yrows*n_cols;

    if(rank_group==0) {
      sendcounts = (int*) malloc( size_group * sizeof(int));
      //displs1 = (int*) malloc( size_group * sizeof(int) );
    }

   // cout << "kron_rows:  " << kron_rows << endl;

    MPI_Gather(&kron_rows, 1, MPI_INT, sendcounts, 1, MPI_INT, 0, comm_group);

    //if(rank_group == 0)
    //  cout << "sendcounts: after gather: " << sendcounts[0] << endl;

    if(rank_group==0) {
    displs1 = (int*) malloc( size_group * sizeof(int) );
    displs1[0] = 0;
      for (int i=1; i<size_group; i++)
        displs1[i] = displs1[i-1] + sendcounts[i-1];
    }
    B_out.setZero(kron_rows);

    MPI_Scatterv(Y.data(), sendcounts, displs1, MPI_FLOAT, B_out.data(), kron_rows, MPI_FLOAT, 0, comm_group);

  }

 // if( rank_group == 0 )
 //   cout << "Passed Scatterv" << endl;

  //if(rank_group == 0)
  //  print4(B_out, "./debug/B_out.txt");

  return B_out;
} 
