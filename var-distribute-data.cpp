#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <iostream>
#include <fstream>
#include <mpi.h>
#define EIGEN_USE_MKL_ALL
#include <eigen3/Eigen/Dense>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "bins.h" 
#include "var-distribute-data.h"

using namespace std;
/*  Only effective if N is much smaller than RAND_MAX */
void shuffle_block(int *array, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

void print_array (int *vec, int rows, char name[]) {

  int leni;
  FILE *fp;
  fp = fopen(name, "w");

  for (leni =0; leni < rows; leni++) {
    fprintf(fp, "%d\n", *(vec + leni));
  }

  fclose (fp);

}

void print_array_int (int vec[], int rows, char name[]) {

  int leni;
  FILE *fp;
  fp = fopen(name, "w");

  for (leni =0; leni < rows; leni++) {
    fprintf(fp, "%d\n", vec[leni]);
  }

  fclose (fp);

}


void print_array_float (float *vec, int rows, char name[]) {

  int leni;
  FILE *fp;
  fp = fopen(name, "w");

  for (leni =0; leni < rows; leni++) {
    fprintf(fp, "%lf\n", *(vec + leni));
  }

  fclose (fp);

}

void var_distribute_data (float *d, int local, int q_rows, int n_rows, int n_cols, int k_rows, float *B_out, int L, int D, MPI_Comm comm_world, MPI_Comm comm_group, int n_readers) 
{

  int i, j;
  size_t sized = (size_t) local * (size_t) (n_cols) * sizeof(float); 

  int rank_world, nprocs_world;
  MPI_Comm_rank(comm_world, &rank_world);
  MPI_Comm_size(comm_world, &nprocs_world);

  int size_group, rank_group;
  MPI_Comm_size(comm_group, &size_group);
  MPI_Comm_rank(comm_group, &rank_group); 


  MPI_Win win;
  MPI_Win_create(d, sized, sizeof(float), MPI_INFO_NULL, comm_world, &win); 

  //#ifndef SIMPLESAMPLE
  int *sample;
  int s = n_rows/L;
  int b = n_rows % L;
  int blks = n_rows - L;

  /*
     number of blks is the number of blocks of L variables to be taken for a n sample dataset. If the number
     of samples are 10 and L = 7. Then there are (10/7) = 1 and 10%7 = 3; So (1 * 7)-3+1 = 5 blocks  
     The samples start from block_ids = [0, 1, 2, 3, 4]; which are shuffled. Then L is added in a loop to 
     create the bootstrap. if shuffled_block_ids = [1, 4, 0, 3, 2], Then the bootstrap row_ids are as follows: 
     - ----------------------------------------------------------------------------------------------------------
     1 4   0 3 2
     _ _   _ _ _
     2 5   1 4 3  
     3 6   2 5 4
     4 7   3 6 5
     5 8   4 7 6
     6 9   5 8 7
     7 10  6 9 8
     --------------------------------------------------------------------------------------------------------
   */

  if (rank_group == 0) {
    int *block_ids;
    block_ids = (int*) malloc (sizeof(int) * blks);
    for (i=0;i<=blks;i++) block_ids[i] = i;
 
    //cout << "blks number from var-dis: " << blks << endl;
 
    //print_array(block_ids, blks, "./debug/block_ids.txt");

    shuffle_block(block_ids, blks); //shuffle blk_ids
    int t;

    //print_array(block_ids, blks, "./debug/block_ids_shuffle.txt");

#ifndef CIRCULARDEPENDENCE
    sample = (int *)malloc((s*L) * sizeof(int) );
#else
    sample = (int *)malloc(sizeof(int) * n_rows)
#endif 

      for (i=0; i<s; i+=D) {
        for(j=0; j<L; j++){
          sample[(i*L)+j] = block_ids[i]+j;

          t = block_ids[i];
        }

      }
  
    //print_array(sample, s*L, "./debug/block_ids_shuffle.txt");

#ifdef CIRCULARDEPENDENCE
    if (b>0){
      int tmp = rand() % b;
      j=0;
      for(i=s*L;i<(s*L)+b;i++) {
        sample[i]=tmp+t+j;
        //printf("sample[%d]=%d\n", i, sample[i]);
        j++;
      }		
    }
#endif

  } else {
    sample = NULL;
  }

  int s_rows[q_rows]; 

  {
    int sendcounts[size_group];
    int displs[size_group];

    for (i=0; i<size_group; i++) {
      int ubound;

#ifndef CIRCULARDEPENDENCE
      bin_range_1D(i, s*L, size_group, &displs[i], &ubound);
      sendcounts[i] = bin_size_1D(i, s*L, size_group);
#else
      bin_range_1D(i, n_rows, size_group, &displs[i], &ubound);
      sendcounts[i] = bin_size_1D(i, n_rows, size_group);
#endif     
    }

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &s_rows, q_rows, MPI_INT, 0, comm_group);

    if (rank_group == 0) free(sample);
  }


  //if(rank_group == 0 ) print_array(s_rows, q_rows, "./debug/srows_0.txt");

  //#endif

  double t = MPI_Wtime();
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

  for (i=0; i<q_rows; i++) {
    int trow = s_rows[i];
    //int target_rank = bin_coord_1D(trow, n_rows, size_group); 
    //int target_rank = bin_coord_1D(trow, n_rows, n_readers);
    //int target_disp = bin_index_1D(trow, n_rows, n_readers) * n_cols;

    //if(rank_group == 0)
    //  cout << "srow: " << i << " " << s_rows[i] << endl;
  
//#ifdef CIRCULARDEPENDENCE
    int target_rank = bin_coord_1D(trow, n_rows, n_readers);
    int target_disp = bin_index_1D(trow, n_rows, n_readers) * n_cols;
//#else
  //  int target_rank = bin_coord_1D(trow, s*L, n_readers);
    //int target_disp = bin_index_1D(trow, s*L, n_readers) * n_cols;
//#endif

    //if (target_disp < 0)
    //   printf("var_dis i: %d\t trow: %d\t n_rows: %d\t size_group: %d\t n_cols: %d\t target_disp: %d\t rank: %d\n", i, trow, n_rows-D, size_group, n_cols, target_disp, rank_group);

    MPI_Get( &B_out[i*n_cols], n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);
  }

  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

  double tmax, tcomm = MPI_Wtime() - t;
  MPI_Reduce(&tcomm, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_world);
  //if (rank_world == 0) {
  //  printf("Comm time: %f (s)\n", tmax);
  //}

  MPI_Win_free(&win);
}


/*  * var_vectorize_response function vectorizes/column stacks all but first D rows of bdata_f. 
    * if N = 101 samples, bdata_f is the blk shuffled closest to N if no circular dependence. So bdata_f.rows()=98
    * In the following function we create samples D to 97. when D=1, 1-97, 
      so 97 enteries leaving out 0th entry i.e., D elements
    * At the last we call s*L, which is bdata_f size if there is no circular dependence. 
*/

void var_vectorize_response(float *d, int local, int yrows, int n_rows, int n_cols, float *B_out, int L, int D, MPI_Comm comm_world, MPI_Comm comm_group)
{

  int i;
  size_t sized = (size_t) local * (size_t) (n_cols) * sizeof(float);  

  int rank_world, nprocs_world;
  MPI_Comm_rank(comm_world, &rank_world);
  MPI_Comm_size(comm_world, &nprocs_world);

  int size_group, rank_group;
  MPI_Comm_size(comm_group, &size_group);
  MPI_Comm_rank(comm_group, &rank_group);


  MPI_Win win;
  MPI_Win_create(d, sized, sizeof(float), MPI_INFO_NULL, comm_world, &win);


  int s = n_rows/L;

#ifndef SIMPLESAMPLE
  int *sample;
  if (rank_group == 0) {
    sample = (int *)malloc((n_rows-D)*sizeof(int)); 
    for (i=0; i<n_rows-D; i++) sample[i]=i; // +D has been commented out.
  } else {
    sample = NULL;
  }

  int s_rows[yrows];

  {
    int sendcounts[size_group];
    int displs[size_group];

    for (i=0; i<size_group; i++) {
      int ubound;
      bin_range_1D(i, n_rows-D, size_group, &displs[i], &ubound);
      sendcounts[i] = bin_size_1D(i, n_rows-D, size_group);
    }

    if (rank_group == 0 )
      print_array(sample, n_rows-D, "./debug/sample_last.dat");

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &s_rows, yrows, MPI_INT, 0, comm_group);

    if(rank_group==0) free(sample); 

  } 

#endif

  double t = MPI_Wtime();
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

  for (i=0; i<yrows; i++) {
#ifdef SIMPLESAMPLE
    int trow = (int) random_at_mostL( (long) n_rows);
#else
    int trow = s_rows[i]; //-1 because now bdata is indexed from 0:N-1 and not from 1:N-2
#endif

#ifndef CIRCULARDEPENDENCE
    int target_rank = bin_coord_1D(trow, s*L, size_group); //n_rows-D has been deleted; s*L is the total number of rows in bdata_f.
    int target_disp = bin_index_1D(trow, s*L, size_group) * n_cols;
#else 
    int target_rank = bin_coord_1D(trow, n_rows, size_group); //n_rows-D has been deleted; s*L is the total number of rows in bdata_f.
    int target_disp = bin_index_1D(trow, n_rows, size_group) * n_cols;
#endif

    MPI_Get( &B_out[i*n_cols], n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);
  }

  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

  /*double tmax, tcomm = MPI_Wtime() - t;
    MPI_Reduce(&tcomm, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_world);
    if (rank_world == 0) {
    printf("Comm time: %f (s)\n", tmax);
    }*/

  MPI_Win_free(&win);


}


void var_generate_Z(float *d, int local, int zrows, int n_rows, int n_cols, float *B_out, int L, int D, MPI_Comm comm_world, MPI_Comm comm_group, int n_readers) 
{
  int i, j;
  size_t sized = (size_t) local * (size_t) (n_cols) * sizeof(float);

  int rank_world, nprocs_world;
  MPI_Comm_rank(comm_world, &rank_world);
  MPI_Comm_size(comm_world, &nprocs_world);

  int size_group, rank_group;
  MPI_Comm_size(comm_group, &size_group);
  MPI_Comm_rank(comm_group, &rank_group);


  MPI_Win win;
  MPI_Win_create(d, sized, sizeof(float), MPI_INFO_NULL, comm_world, &win);	

  int s = n_rows/L;

#ifdef CIRCULARDEPENDENCE
  int sample_size = n_rows-D;
#else
  int sample_size = (s*L) - D;
#endif

//#ifndef SIMPLESAMPLE
  int *sample;
  if (rank_group == 0) {
    sample = (int *)malloc(sample_size * sizeof(int));
    /*for (i=0; i<n_rows-D; i++)
      for (j=0;j<D;j++)
        sample[(i*D)+j]=i+D+j;*/
    for(i=0; i<sample_size;i++)
      sample[i] = i+D;

  } else {
    sample = NULL;
  }

  //int yrows = bin_size_1D(rank_group, (n_rows-D)*D, nprocs_group)
  int s_rows[zrows];

  {
    int sendcounts[size_group];
    int displs[size_group];

    for (i=0; i<size_group; i++) {
      int ubound;
      bin_range_1D(i, sample_size, size_group, &displs[i], &ubound);
      sendcounts[i] = bin_size_1D(i, sample_size, size_group);
    }

    //if (rank_group == 0 )
     // print_array(sample, sample_size, "./debug/sample_z.dat");

    MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &s_rows, zrows, MPI_INT, 0, comm_group);

    if(rank_group==0) free(sample);

  }

//#endif 


  double t = MPI_Wtime();
  MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

  for (i=0; i<zrows; i++) {
#ifdef SIMPLESAMPLE
    int trow = (int) random_at_mostL( (long) n_rows);
#else
    int trow = s_rows[i]; //-1 because now bdata is indexed from 0:N-1 and not from 1:N-2
#endif

#ifdef CIRCULARDEPENDENCE
    int target_rank = bin_coord_1D(trow, n_rows, n_readers);  
    int target_disp = bin_index_1D(trow, n_rows, n_readers) * n_cols;
#else
    int target_rank = bin_coord_1D(trow, s*L, n_readers);
    int target_disp = bin_index_1D(trow, s*L, n_readers) * n_cols;
#endif

    //if (target_disp < 0)
    //  printf("var_gen i: %d\t trow: %d\t n_rows: %d\t size_group: %d\t n_cols: %d\t target_disp: %d\t rank: %d\n", i, trow, n_rows-D, size_group, n_cols, target_disp, rank_group);

    MPI_Get( &B_out[i*n_cols], n_cols, MPI_FLOAT, target_rank, target_disp, n_cols, MPI_FLOAT, win);
  }

  MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

  MPI_Win_free(&win);
}
