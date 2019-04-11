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

void print3( MatrixXf m, string name )
{
  std::ofstream file(name);
  if (file.is_open())
  {
    file  << m << '\n';
  }

}

MatrixXf var_kron (float* d, int local, int yrows, int n_rows, int n_cols, int D, MPI_Comm comm_r,  MPI_Comm comm_world, MPI_Comm comm_group, int n_readers)
{
  int i, j;
  size_t sized;
  if (MPI_COMM_NULL != comm_r)
    sized = (size_t) local * (size_t) (n_cols) * sizeof(float);
  else
    sized = 0;
    

  int rank_world, nprocs_world;
  MPI_Comm_rank(comm_world, &rank_world);
  MPI_Comm_size(comm_world, &nprocs_world);

  int size_group, rank_group;
  MPI_Comm_size(comm_group, &size_group);
  MPI_Comm_rank(comm_group, &rank_group);


  MPI_Win win;
  MPI_Win_create(d, sized, sizeof(float), MPI_INFO_NULL, comm_world, &win);

  if(rank_world==0)
      cout << "passed create" << endl;

 int *sample;
   if (rank_group==0) {
        sample = (int *)malloc((n_rows-D)* n_cols * sizeof(int));
        for (int i=0; i<(n_rows-D)*n_cols;i++) sample[i] = i;
        //print_array1(sample, (n_rows-D)*n_cols, "./debug/sample_var.txt");
   } else {
    sample = NULL;
  }

  
  if(rank_world==0)
      cout << "passed create" << endl;

  //TODO: Fix the bin_range1D function. It is given out incorrect displs. 
  // The current implementation works fine. Please use this and do not uncomment it.  
  // - Mahesh

  int srows[yrows];
  {
    //int sendcounts[size_group];
    //int displs[size_group];
    int *sendcounts;
    int *displs;

    if (rank_group==0) {
      sendcounts = (int*) malloc( size_group * sizeof(int));
      displs = (int*) malloc( size_group * sizeof(int));  
    }
  
    MPI_Gather(&yrows, 1, MPI_INT, sendcounts, 1, MPI_INT, 0, comm_group); 
  
    if(rank_world==0)
      cout << "passed gather" << endl;

    if(rank_group==0) { 
      displs[0] = 0;
      /*for (i=0; i<size_group; i++) {
        //int ubound;
      // bin_range_1D(i, (n_rows-D)* n_cols, size_group, &displs[i], &ubound);
        sendcounts[i] = bin_size_1D(i, (n_rows-D), size_group) * n_cols;
      }*/ 
      for(i=1; i<size_group; i++)
        displs[i] = displs[i-1] + sendcounts[i-1];
    }

    /*if(rank_world==0) {
      for (i=0; i<size_group; i++) {
        cout << "sendcnt: " << i << " " << sendcounts[i];
        cout << "displs: " << i << " " << displs[i];
      }
    }*/ 

    if(rank_world==0)
      cout << "before scatterv" << endl;

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &srows, yrows, MPI_INT, 0, comm_group);
  
      if(rank_world==0)
      cout << "passed scatterv" << endl;

    /*if ( rank_group == 0 )  {
      print_array1(sample, (n_rows-D)*n_cols, "./debug/sample_var.dat"); 
      print_array1(sendcounts, size_group, "./debug/sendcnts_var.dat");
      print_array1(displs, size_group, "./debug/displs_var.dat");
    }*/
    /*if ( rank_group == 0 ) print_array1(srows, yrows, "./debug/srows_0.dat");
    if ( rank_group == 1 ) print_array1(srows, yrows, "./debug/srows_1.dat");
    if ( rank_group == 2 ) print_array1(srows, yrows, "./debug/srows_2.dat");
    if ( rank_group == 3 ) print_array1(srows, yrows, "./debug/srows_3.dat");*/ 

    if(rank_group==0) free(sample);
  }


  //if(rank_group == 0) cout << "n_cols in here: " << n_cols << endl;  

  double t = MPI_Wtime();
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);
  
  MatrixXf B_out(yrows, D*n_cols*n_cols);
  B_out.setZero();  
  VectorXf get_vector(n_cols);
 
  if(rank_world==0)
      cout << "passed before loop" << endl;
  
  for (i=0; i<yrows; i++) {
    get_vector.setZero();
#ifdef SIMPLESAMPLE
    int trow = (int) random_at_mostL( (long) n_rows);
#else
    int trow = srows[i] % (n_rows-D);
#endif
    //int target_rank = bin_coord_1D(trow, (n_rows-D), nprocs_world);
    int target_rank = bin_coord_1D(trow, (n_rows-D), n_readers);
    int target_disp = bin_index_1D(trow, (n_rows-D), n_readers) * n_cols;
    int col_disp = srows[i] / (n_rows-D);  
    MPI_Get(get_vector.data(), n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);
    //MPI_Get(B_out.block(i, n_cols*col_disp, 1, n_cols).data(), n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);

    /*if( get_vector.isZero() && rank_group == 68) {
        cout << i << "th row of rank 68 is empty" << endl;
        cout << " trow:" << trow << "target_rank:" << target_rank << "col_disp:" << col_disp << endl << endl; 
        print3(get_vector, "./debug/get_vector_" + to_string(rank_group) + "_" + to_string(i) + ".txt");
    }*/
  
    for(j=0; j<n_cols;j++)
     B_out(i, j+(n_cols*col_disp)) = get_vector(j);
  }
  
    if(rank_world==0)
      cout << "passed after loop" << endl;
  /*for(i=0; i<yrows; i++) { 
    if( B_out.row(i).isZero())
      cout << "from var_kron row: " << i << " rank: " << rank_group << endl;
      print3(B_out.row(i), "./debug/B_out_" + to_string(rank_group) + ".txt");

    }*/
 
  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
  MPI_Win_free(&win);
	
  return B_out;
}
