#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <boost/tuple/tuple.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include <mpi.h>
#include <assert.h>
#include <cmath>
#define EIGEN_USE_MKL_ALL
#include "lasso.h"
#include "structure.h"
#include "manage-data.h"
#include "UoI_VAR.h"
#include "bins.h"
#include "var_kron.h"
#include "var_vectorize.h"
#include "var-distribute-data.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>



using namespace std;
using namespace Eigen;

//########################
//Funcs used in UoI_VAR
//########################
float not_NaN (float x) {if (!isnan(x)) return x; else return 0;}
void print( MatrixXf , string );
void print_int( VectorXi, string );
void print_vint( vector<int>, string );
VectorXf logspace ( int, int, int );
float pearson ( VectorXf, VectorXf );
float explained_variable_R2(VectorXf, VectorXf);
MatrixXf lasso_sweep (float *,  int, int, VectorXf, int, int, int, int, int, bool, int, float, float, float, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, int);
//MatrixXf CreateZ ( MatrixXf, int );
MatrixXf stack_Z (float* , int , int , int) ;
MatrixXf KroneckerProduct( MatrixXf, MatrixXf, int );
VectorXf VectorizeY( MatrixXf );
MatrixXf CreateSupport(int, int, int, MatrixXf);
boost::tuple<MatrixXf, MatrixXf, vector<int> > ApplySupport(MatrixXf& , MatrixXf&, VectorXf);
MatrixXf ApplySupport_test(MatrixXf& , vector<int> , int);
VectorXf median (MatrixXf );
//########################
//Funcs definitions
//########################


VectorXf UoI_VAR(INIT *init) 
{
  //Initialize MPI for UoI_Lasso processes the processes
  int rank, nprocs;
  MPI_Comm world_comm = init->comm;
  MPI_Comm_size(world_comm, &nprocs);
  MPI_Comm_rank(world_comm, &rank);
  MPI_Group world_group;
  MPI_Comm_group(world_comm, &world_group);


  int N; //dataset dimension N rows are samples; p columns are features. 
  int p;

  double total_start, total_end;

  if (rank == 0)
    total_start = MPI_Wtime();


  if (rank == 0)
  {
    //extract model dimensions from design matrix
    N = get_rows(init->Infile, init->data_mat);
    p = get_cols(init->Infile, init->data_mat);
  }

  MPI_Bcast(&N, 1, MPI_INT, 0, world_comm);
  MPI_Bcast(&p, 1, MPI_INT, 0, world_comm);

  if ( (rank == 0 ) && init->debug) {
    cout << "nprocs: " << nprocs << endl;
    cout << "n_samples: " << N << endl;
    cout << "n_features: " << p << endl;
  }
  //check for processors limits
  if ( (rank == 0 ) && (init->n_readers > N) )
  {
    printf("must have n_readers (cores used for reading) < n_samples in the dataset \n");
    fflush(stdout);
    MPI_Abort(world_comm, 3);
  }

  if ( ( rank == 0) && ( init->n_groups > nprocs ) )
  {
    printf("must have ngroups < nprocs \n");
    fflush(stdout);
    MPI_Abort(world_comm, 4);
  }

  //int local_rows = bin_size_1D(rank, N, nprocs);
  //MatrixXf data;
  //Map<Matrix<float, Dynamic, Dynamic, RowMajor>> data( get_matrix(local_rows, p, N, world_comm, rank, init->data_mat, init->Infile), local_rows,p);
  /*Map<MatrixXf> data( get_matrix(local_rows, p, N, world_comm, rank, init->data_mat, init->Infile), local_rows,p);

    if ( (rank == 0 ) && init->verbose)
    cout << "(1) Loaded data.\n" << N << " samples with " << p << " features."   << endl;

    if ( ( rank == 0 ) && init->debug)
    {
    print( data, "./debug/data.txt" );
    }*/


  //------------------------------------------------
  //create color with init.n_groups for level 1 parallelization
  int color = bin_coord_1D(rank, nprocs, init->n_groups);
  MPI_Comm comm_g;
  MPI_Comm_split(world_comm, color, rank, &comm_g);

  int nprocs_g, rank_g;
  MPI_Comm_size(comm_g, &nprocs_g);
  MPI_Comm_rank(comm_g, &rank_g);
  MPI_Group L1_group;
  MPI_Comm_group(comm_g, &L1_group);

  if( (rank==0) ) {
    if ( nprocs_g < init->n_readers ) {
      printf("must have nprocs_per_bootstrap >= n_readers (usually is order of 1-nprocs_per_bootstrap)\t set --n_readers<=%d\n",nprocs_g);
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
  }

  //create color iwth init.n_minigroups for level 2 parallelism
  color = bin_coord_1D(rank_g, nprocs_g, init->n_minigroups);
  MPI_Comm comm_mg;
  MPI_Comm_split(comm_g, color, rank_g, &comm_mg);

  int nprocs_mg, rank_mg;
  MPI_Comm_size(comm_mg, &nprocs_mg);
  MPI_Comm_rank(comm_mg, &rank_mg);

  //create L1 and L2 root processes as separate groups.
  //This is for easier output management.

  //Level-1 root ranks group creation.
  int l1_root_rank = -1, l1_root_size = -1;

  VectorXi l1_roots(init->n_groups);
  int root = 0;

  for (int i=0; i<init->n_groups; i++)
  {
    root = i * nprocs_g;
    l1_roots(i) = root;
  }

  MPI_Group l1_root_group;
  MPI_Group_incl(world_group, init->n_groups, l1_roots.data(), &l1_root_group);
  MPI_Comm L1_roots_comm;
  MPI_Comm_create_group(world_comm, l1_root_group, 0, &L1_roots_comm);

  if (MPI_COMM_NULL != L1_roots_comm)
  {
    MPI_Comm_rank(L1_roots_comm, &l1_root_rank);
    MPI_Comm_size(L1_roots_comm, &l1_root_size);
  }

  //Level-2 root ranks group creation.
  int l2_root_rank = -1, l2_root_size = -1;

  VectorXi l2_roots(init->n_minigroups);
  root = 0;

  for (int i=0; i<init->n_minigroups; i++)
  {
    root = i * nprocs_mg;
    l2_roots(i) = root;
  }

  MPI_Group l2_root_group;
  MPI_Group_incl(L1_group, init->n_minigroups, l2_roots.data(), &l2_root_group);
  MPI_Comm L2_roots_comm;
  MPI_Comm_create_group(comm_g, l2_root_group, 0, &L2_roots_comm);

  if (MPI_COMM_NULL != L2_roots_comm)
  {
    MPI_Comm_rank(L2_roots_comm, &l2_root_rank);
    MPI_Comm_size(L2_roots_comm, &l2_root_size);
  }

  //MPI_Barrier(world_comm);

  //Create loader groups in every bootstrap. This to make use of the parallelism, use this
  // until a good strategy is available.

  VectorXi readers(init->n_readers);

  if(init->n_readers == 1)
    readers(0) = 0;
  else
    readers.setLinSpaced(init->n_readers, 0, init->n_readers);


  MPI_Group reader_group;
  MPI_Group_incl(L1_group, init->n_readers, readers.data(), &reader_group);
  MPI_Comm comm_r;
  MPI_Comm_create_group(comm_g, reader_group, 0, &comm_r);

  int rank_r = -1, nprocs_r = -1;

  if (MPI_COMM_NULL != comm_r)
  {
    MPI_Comm_rank(comm_r, &rank_r);
    MPI_Comm_size(comm_r, &nprocs_r);
  }

  //Completed creating all logical processes split communicators
  //Now load the data. 
  double load_s, load_e;

  if(rank==0) load_s = MPI_Wtime();

  int local_rows;

  if(MPI_COMM_NULL != comm_r)
    local_rows = bin_size_1D(rank_r, N, nprocs_r);
  else
    local_rows = 0;


  //Now load the data into the reader cores.
  //MatrixXf data;
  float *data;
  //if(MPI_COMM_NULL != comm_r) 
  //  Map<MatrixXf> data( get_matrix(local_rows, p, N, comm_r, rank_r, init->data_mat, init->Infile), local_rows,p);

  if(rank ==0 ) load_s = MPI_Wtime();

  if(MPI_COMM_NULL != comm_r) {
    data = (float *)malloc(local_rows * p * sizeof(float));
    data = get_matrix(local_rows, p, N, comm_r, rank_r, init->data_mat, init->Infile);
  }

  if( rank == 0 ) {load_e = MPI_Wtime() - load_s; cout << "Load time: " << load_e << " (s)" << endl;}

  if ( (rank == 0 ) && init->verbose)
    cout << "(1) Loaded data " << N << " samples and " << p << " features with " << init->n_readers << " cores."   << endl;

  if ( ( rank == 0 ) && init->debug)
  { 
    Map<MatrixXf> data_print(data, local_rows, p);
    cout << "data size: " << data_print.rows() << " " << data_print.cols() << endl;
    print( data_print, "./debug/data.txt" );
  }

  if ( ( rank == 0 ) && init->debug) {cout << "All parameters are set for selection."  << endl;}

  //All the parameters are set. 
  //--------------------------------------------------------------

  int bootstrap_subsample=0;

#ifndef CIRCULARDEPENDENCE
  bootstrap_subsample = N - (N%(init->L));
#else
  bootstrap_subsample = N; 
#endif

  //if(init->debug)
  //  cout << "bootstrap variable: " << bootstrap_subsample << "  rank: " << rank << endl;

  int qrows = bin_size_1D(rank_r, bootstrap_subsample, nprocs_r);

  //qrows = bin_size_1D(rank_mg, bootstrap_subsample, nprocs_mg); //rows per distributed bootstrap.
  //qrows = local_rows;


  if(rank==0 && init->debug) 
    cout << "qrows before calling: " << qrows << endl;

  //Create lambda vector
  VectorXf lambda(init->n_lambdas);
  VectorXf lambda_dis(init->n_lambdas/init->n_minigroups );

  if ( rank_g == 0 )
  {
    if (init->n_lambdas == 1)
      lambda.setOnes();
    else
      lambda = logspace(-2, 1, init->n_lambdas);
  }

  if(rank_g == 0 && init->debug) print(lambda, "./debug/lambda_vec.txt");

  MPI_Bcast(lambda.data(), init->n_lambdas, MPI_FLOAT, 0, world_comm);

  if ( rank_mg == 0 )
    lambda_dis = lambda.segment(l2_root_rank*(init->n_lambdas/init->n_minigroups), init->n_lambdas/init->n_minigroups);

  //if (MPI_COMM_NULL != L2_roots_comm)
  MPI_Bcast(lambda_dis.data(), lambda_dis.size(), MPI_FLOAT, 0, comm_mg);

  if ( (rank == 0) && init->verbose)
    cout << "(2) Beginning model selection." << endl;

  int threshold = (init->selection_thres_frac * (init->n_boots_sel-1)); 

  //#######################
  //### Model Selection ###
  //#######################

  double sel_time_s, sel_time_e;
  MatrixXf estimates_selection;

  if (rank == 0) {sel_time_s = MPI_Wtime();}
  estimates_selection = lasso_sweep (data, local_rows, p, lambda_dis, qrows, N,  init->L, init->D, 
      (init->n_boots_sel/init->n_groups), init->debug, init->max_iter, init->reltol, 
      init->abstol, init->rho, world_comm, comm_g, comm_mg, comm_r, init->n_readers);

  if (rank == 0)
  {
    sel_time_e = MPI_Wtime() - sel_time_s;
    cout << "Total Selection time: " << sel_time_e << "(s)" << endl;

  }
  if ( (rank == 0) && init->debug ) // && init->debug
    print( estimates_selection, "./debug/estimates_selection.txt"); 

  //create support matrix storage
  double sup_time_s, sup_time_e;
  MatrixXf supports_(init->n_lambdas, p*p);


  if ( rank_mg == 0) 
  {
    sup_time_s = MPI_Wtime();
    supports_ = CreateSupport(init->n_lambdas, init->n_boots_sel, threshold, estimates_selection);
    sup_time_e = MPI_Wtime() - sup_time_s;
    cout << "Supports creation time: " << sup_time_e << "(s)" << endl;
  }

  estimates_selection.resize(0,0);

  if ( (rank == 0) && init->debug ) //&& init->debug
    print( supports_, "./debug/supports_.txt");

  MPI_Bcast(supports_.data(), init->n_lambdas*p*p, MPI_FLOAT, 0, world_comm);

  /*if (init->debug)
    {
    MPI_Barrier(world_comm);
    MPI_Abort(world_comm, 1);

    }*/ 

  //--TODO Completed till here.  -- Compiles without errors

  //Set all the estimation bootstrap MPI parameters before Estimation step

  color = bin_coord_1D(rank, nprocs, init->n_est);
  MPI_Comm comm_e;
  MPI_Comm_split(world_comm, color, rank, &comm_e);

  int nprocs_e, rank_e;
  MPI_Comm_size(comm_e, &nprocs_e);
  MPI_Comm_rank(comm_e, &rank_e);
  MPI_Group L1_group_e;
  MPI_Comm_group(comm_e, &L1_group_e);

  //create color iwth init.n_minigroups for level 2 parallelism
  color = bin_coord_1D(rank_e, nprocs_e, init->n_miniest);
  MPI_Comm comm_me;
  MPI_Comm_split(comm_e, color, rank_e, &comm_me);

  int nprocs_me, rank_me;
  MPI_Comm_size(comm_me, &nprocs_me);
  MPI_Comm_rank(comm_me, &rank_me);

  //create L1 and L2 root processes as separate groups.
  //This is for easier output management.

  //Level-1 root ranks group creation.
  int l1_root_rank_e = -1, l1_root_size_e = -1;

  VectorXi l1_roots_e(init->n_est);
  int root_e = 0;

  for (int i=0; i<init->n_est; i++)
  {
    root_e += i * nprocs_e;
    l1_roots_e(i) = root_e;
  }

  MPI_Group l1_root_group_e;
  MPI_Group_incl(world_group, init->n_est, l1_roots_e.data(), &l1_root_group_e);
  MPI_Comm L1_roots_comm_e;
  MPI_Comm_create_group(world_comm, l1_root_group_e, 0, &L1_roots_comm_e);

  if (MPI_COMM_NULL != L1_roots_comm_e)
  {
    MPI_Comm_rank(L1_roots_comm_e, &l1_root_rank_e);
    MPI_Comm_size(L1_roots_comm_e, &l1_root_size_e);
  }

  //Level-2 root ranks group creation.
  int l2_root_rank_e = -1, l2_root_size_e = -1;

  VectorXi l2_roots_e(init->n_miniest);
  root_e = 0;

  for (int i=0; i<init->n_miniest; i++)
  {
    root_e += i * nprocs_me;
    l2_roots_e(i) = root_e;
  }

  MPI_Group l2_root_group_e;
  MPI_Group_incl(L1_group_e, init->n_miniest, l2_roots_e.data(), &l2_root_group_e);
  MPI_Comm L2_roots_comm_e;
  MPI_Comm_create_group(comm_e, l2_root_group_e, 0, &L2_roots_comm_e);

  if (MPI_COMM_NULL != L2_roots_comm_e)
  {
    MPI_Comm_rank(L2_roots_comm_e, &l2_root_rank_e);
    MPI_Comm_size(L2_roots_comm_e, &l2_root_size_e);
  }

  //MPI_Barrier(world_comm);

  if ( ( rank == 0 ) && init->verbose) {cout << "All parameters are set for estimation." << endl;}

  //#######################
  //### Model Estimation ##
  //#######################

  //MPI_Barrier(world_comm);

  int size_s = (N-(init->L)+2) * (init->L); 

  //float *data_f;
  //data_f = (float*)malloc(local_rows * p * sizeof(float) );
  //Map<MatrixXf> (data_f, data.rows(), data.cols()) = data; 
  float *bdata_f; 
  MatrixXf Z_mx_1, Z_mx_2, X_train, X_test; 
  VectorXf Y_train, Y_test, z, r, y_hat, y_true;
  VectorXf mspe(init->n_lambdas/init->n_miniest);
  //VectorXf R2(init->n_lambdas/init->n_miniest);

  float *Z_mx_e, *Z_stacked, *X_, *Y_, *Y_t;

  //sizes for distribution
  int z_rows;

  if(MPI_COMM_NULL != comm_r)
    z_rows = bin_size_1D(rank_r, (bootstrap_subsample-init->D)*init->D, nprocs_r);

  int y_rows = bin_size_1D(rank_e, (bootstrap_subsample-init->D), nprocs_e);

  int kron_rows  = bin_size_1D(rank_e, (bootstrap_subsample-init->D)*p, nprocs_e);

  int boots_est = init->n_boots_est/init->n_est;   
  int lambs = (init->n_lambdas/init->n_miniest); 
  int p2 = p*p;

  MatrixXf lasso_estimates(lambs,p2);

  //Containers to store bootstrap estimates and the best estimates
  VectorXf est_(p2);
  //VectorXf whichbest(boots_est);
  MatrixXf scores(init->n_boots_est, init->n_lambdas);
  MatrixXf best_estimates(boots_est, p2);

  double time_s, time_e, time_w_s, time_w_e, lasso_est_comm, lasso_comm, time_las_e, time_las_s; 

  if (rank == 0)
    time_s = MPI_Wtime();

  int Y_mean;

  //containers for first rearrangement
  float *bdata_f_1, *Z_mx_e_1, *Z_stacked_1;

  //containers for second rearragement
  float *bdata_f_2, *Z_mx_e_2, *Z_stacked_2; 

  if (MPI_COMM_NULL != comm_r) {  
    //initialize containers for 1st rearrangement
    bdata_f_1 = (float *)malloc(qrows * p * sizeof(float) );
    Z_mx_e_1 = (float *)malloc(z_rows * p * sizeof(float));
    Z_stacked_1 = (float *)malloc(z_rows * init->D*p * sizeof(float));

    //initialize containers for 2nd rearrangement
    bdata_f_2 = (float *)malloc(qrows * p * sizeof(float) );
    Z_mx_e_2 = (float *)malloc(z_rows * p * sizeof(float));
    Z_stacked_2 = (float *)malloc(z_rows * init->D*p * sizeof(float));

  }

  double time7, time8;
  for (int bootstraps=0; bootstraps < boots_est; bootstraps++)
  {

    //first rearrangement to create train datasets
    //bdata_f = (float *)malloc(qrows * p * sizeof(float) );

    //  if( rank == 0 && bootstraps == 0 )
    //    cout << "Inside estimation bootstrap loop" << endl; 

    if (MPI_COMM_NULL != comm_r) {
      var_distribute_data(data, qrows, qrows, N, p, size_s, bdata_f_1, init->L, init->D, comm_r, comm_r, init->n_readers );
      var_distribute_data(data, qrows, qrows, N, p, size_s, bdata_f_2, init->L, init->D, comm_r, comm_r, init->n_readers );
    }
    //Map<MatrixXf>bdata(bdata_f, qrows, p);
  
    //if( rank == 0 && bootstraps == 0 )
    //    cout << "After estimation var_distribute" << endl;

    if ( rank == 0 && bootstraps == 0 && init->debug) {
      Map<MatrixXf> bdata(bdata_f_1, qrows, p);
      print(bdata, "./debug/bdata_0_est.txt");

    }


  //if( rank == 0 && bootstraps == 0 )
  //      cout << "After bdata_print" << endl;

    // Generate Z_mx matrix
    //Z_mx_e = (float *)malloc(z_rows * p * sizeof(float));
 
    double time3_est;
    if (rank == 0 && bootstraps == 0 )
      time3_est = MPI_Wtime();
 
    if (MPI_COMM_NULL != comm_r) {
      var_generate_Z(bdata_f_1, qrows, z_rows, N, p, Z_mx_e_1, init->L, init->D, comm_r, comm_r, init->n_readers);
      var_generate_Z(bdata_f_2, qrows, z_rows, N, p, Z_mx_e_2, init->L, init->D, comm_r, comm_r, init->n_readers);
    }

    if ( rank== 0 && bootstraps == 0 )
    {
      double time4 = MPI_Wtime() - time3_est;
      cout << "var_gen_Z_time: " << time4 << "(s)" << endl;
    }
  

   // if( rank == 0 && bootstraps == 0 )
   //     cout << "After estimation var_generate_Z" << endl;
    //Z_mx = stack_Z(Z_mx_e, z_rows, p, init->D);


    double time4_est;
    if (rank == 0 && bootstraps == 0 )
      time4_est = MPI_Wtime();


    if(MPI_COMM_NULL != comm_r) {
      Z_mx_1 = stack_Z(Z_mx_e_1, z_rows, p, init->D);
      //cout << "completed stack" << endl;
      Map<MatrixXf>(Z_stacked_1, Z_mx_1.rows(), Z_mx_1.cols()) = Z_mx_1;

      Z_mx_2 = stack_Z(Z_mx_e_2, z_rows, p, init->D);
      Map<MatrixXf>(Z_stacked_2, Z_mx_2.rows(), Z_mx_2.cols()) = Z_mx_2;
    }

     if ( rank == 0 && bootstraps == 0 )
    {
      double time5 = MPI_Wtime() - time4_est;
      cout << "var_stack_Z_time: " << time5 << "(s)" << endl;
    }

   // if( rank == 0 && bootstraps == 0 )
   //     cout << "After estimation var_stacked" << endl;

    if ( rank == 0 && bootstraps == 0 && init->debug )
      print(Z_mx_1, "./debug/Z_mx_0_est.txt");

    //Z_stacked = (float *)malloc(Z_mx.rows() * Z_mx.cols() * sizeof(float));
    //Map<MatrixXf>(Z_stacked, Z_mx.rows(), Z_mx.cols()) = Z_mx;
    //int kron_rows = y_rows * p;

     double time6_est, time7_est, time8_est;
    if (rank == 0)
      time6_est = MPI_Wtime();
  
    //X_train = var_kron (Z_stacked, Z_mx.rows(), kron_rows, bootstrap_subsample, p, init->D, world_comm, comm_e);
    X_train = var_kron (Z_stacked_1, Z_mx_1.rows(), kron_rows, bootstrap_subsample, p, init->D, comm_r, world_comm, comm_e, init->n_readers);

    //double time7, time8;
    if(rank==0) {
      time7 += MPI_Wtime() - time6_est;
      if(bootstraps == 0)
        cout << "var_kron time: " << time7 << "(s)" << endl;
    }
        

   if (rank == 0)
      time7_est = MPI_Wtime(); 
   // if( rank == 0 && bootstraps == 0 )
   //     cout << "After estimation var_kron train" << endl;

    Y_train = var_vectorize (bdata_f_1, qrows, kron_rows, bootstrap_subsample, p , init->D, comm_r,  world_comm, comm_e, init->n_readers); 

    if(rank==0) {
      time8 += MPI_Wtime() - time7_est;
      if(bootstraps == 0)
        cout << "var_kron time: " << time8 << "(s)" << endl;
    }
   // if( rank == 0 && bootstraps == 0 )
   //     cout << "After estimation var_vec train" << endl;
    //Y_ = (float *)malloc(kron_rows * sizeof(float));
    //var_vectorize_response(bdata_f, qrows, y_rows, bootstrap_subsample, p, Y_, init->L, init->D, world_comm, comm_e);
    //Map<VectorXf> Y_train(Y_, kron_rows); 


    if ( rank == 0 && bootstraps == 0 && init->debug )
      print(Y_train, "./debug/Y_train_est.txt");

    if ( rank == 0 && bootstraps == 0 && init->debug )
      print(X_train, "./debug/X_train_est.txt");

    if (rank == 0)
      time6_est = MPI_Wtime();
    //second rearragement to create test datasets
    X_test = var_kron (Z_stacked_2, Z_mx_2.rows(), kron_rows, bootstrap_subsample, p, init->D, comm_r, world_comm, comm_e, init->n_readers);
    
    if(rank==0) {
      time7 += MPI_Wtime() - time6_est;
      //if(bootstraps == 0)
        //cout << "var_kron time: " << time7 << "(s)" << endl;
    }


   // if( rank == 0 && bootstraps == 0 )
   //     cout << "After estimation var_kron test" << endl;

    if ( rank == 0 && bootstraps == 0 && init->debug )
      print(X_test, "./debug/X_test.txt");

    if (rank == 0)
      time7_est = MPI_Wtime();

    Y_test = var_vectorize (bdata_f_2, qrows, kron_rows, bootstrap_subsample, p , init->D, comm_r,  world_comm, comm_e, init->n_readers);

    if(rank==0) {
      time8 += MPI_Wtime() - time7_est;
     // if(bootstraps == 0)
     //   cout << "var_kron time: " << time8 << "(s)" << endl;
    }

   // if( rank == 0 && bootstraps == 0 && init->debug)
   //     cout << "After estimation var_vec test" << endl;

    for (int lambda_idx=0; lambda_idx < (init->n_lambdas/init->n_miniest); lambda_idx++)
    {

      est_.setZero();
      //lasso_estimates.setZero();
      lasso_comm = 0.0; 
      vector<int> supportids;

      //     if( rank == 0 )
      //       cout << " passed est_set" << endl;

      if (rank == 0)
        time_las_s = MPI_Wtime();

      MatrixXf X_recon, X_test_recon;
      VectorXf est_test;
       
      //Apply Support
      boost::tie(X_recon, X_test_recon, supportids) = ApplySupport(X_train, X_test, supports_.row(lambda_idx)); 
    
     //     if( rank == 0 )
     //       cout << " passed apply support" << endl;

      //if (rank == 0 && init->debug)  cout << "X_recon size: " << X_recon.rows() << " ," << X_recon.cols() << endl;

      //X_test_recon = ApplySupport_test(X_test, supportids, p2);
      //The following block finds the global mean of the Y vector for centering/

      if ( rank == 0 && bootstraps == 0 && lambda_idx==0 && init->debug ) {
        print(X_recon, "./debug/X_recon_est.txt");
        print(X_test_recon, "./debug/X_test_recon.txt");
        print_vint(supportids, "./debug/supportids.txt");
      }

      //if(rank==0 && bootstraps==0 && lambda_idx == 5)
      //    print_vint(supportids, "./debug/supportids_5.txt");

      //Compute OLS
      if( supportids.size() != 0 ) 
      {
        //if(rank==0 && bootstraps==0 && lambda_idx == 5)
        //    cout << "I am inside if-estimation" << endl;    
        boost::tie(z,lasso_comm) = lasso(X_recon, Y_train, 0.0, init->max_iter, init->reltol, init->abstol, init->rho, comm_me);

        if ( rank == 0 && bootstraps == 0 && lambda_idx==0 && init->debug ) // && init->debug
          print(z, "./debug/z_est.txt");

        for (int i = 0; i<supportids.size(); i++)
          est_(supportids[i]) = z(i); 
      

       // if( rank == 0 )
       //   cout << " passed est_" << endl;

        if ( rank == 0 && bootstraps == 0 && lambda_idx==0 && init->debug) // && init->debug
          print(est_, "./debug/est_estimation.txt");

        if (rank == 0) 
          time_las_e += MPI_Wtime() - time_las_s;

        //z_est 
        lasso_est_comm += lasso_comm;

        lasso_estimates.row(lambda_idx) = est_;
        y_hat = X_test * est_;
  
        float mspe_val;
        VectorXf Y_m, Y_hat, Y_Test;


        //Since for each lambda distributed the estimates are different and X_test and Y_test are the same
        // we gather them to the l1 root to find the root mspe at the l1 root (1st level parallel).  
        {
          VectorXf Y_send = (Y_test-y_hat).array().square();
          int *displs = NULL;
          int *sizecounts = NULL;
          int recvsize=0; 

          if (rank_e == 0)
            sizecounts = (int*) malloc( nprocs_e * sizeof(int));

          int Y_size = Y_send.size();

          MPI_Gather(&Y_size, 1, MPI_INT, sizecounts, 1, MPI_INT, 0, comm_e);

          if (rank_e == 0) {
            int totlen = 0;
            displs = (int*) malloc( nprocs_e * sizeof(int) );

            displs[0] = 0;
            totlen += sizecounts[0];

            for (int i=1; i<nprocs_e; i++) {
              totlen += sizecounts[i];
              displs[i] = displs[i-1] + sizecounts[i-1];
            }

            for (int i=0; i<nprocs_e; i++)
              recvsize += sizecounts[i];

            if (rank_e == 0) {
                Y_m.setZero(recvsize);
                Y_hat.setZero(recvsize);
                Y_Test.setZero(recvsize); 
            }
          }
  
          MPI_Gatherv(y_hat.data(), y_hat.size(), MPI_FLOAT, Y_hat.data(), sizecounts, displs, MPI_FLOAT, 0, comm_e);
          MPI_Gatherv(Y_test.data(), Y_test.size(), MPI_FLOAT, Y_Test.data(), sizecounts, displs, MPI_FLOAT, 0, comm_e);
          MPI_Gatherv(Y_send.data(), Y_send.size(), MPI_FLOAT, Y_m.data(), sizecounts, displs, MPI_FLOAT, 0, comm_e);  

          if (rank_e == 0)
              mspe_val = Y_m.mean(); 

          //MPI_Bcast(&mspe_val, 1, MPI_FLOAT, 0, comm_e);       


        }

        if(rank_e == 0) {
          mspe(lambda_idx)  = mspe_val;
          //float r2 = explained_variable_R2(Y_hat, Y_Test);
          //float r2 = pearson(Y_hat, Y_Test);
          //scores(bootstraps,lambda_idx) = r2;
        }
        
      }   
    }

    if( rank == 0 && bootstraps == 0 ){
      print(mspe, "./debug/mspe.txt");
     // print(lasso_estimates, "./debug/lasso_estimates.txt");
    }
  
    if ( rank == 0 && bootstraps==0 && init->debug ) {
      print(mspe, "./debug/mspe.txt");
      //print(mspe.unaryExpr(ptr_fun(not_NaN)), "./debug/mspe_notnan.txt"); 
      print(lasso_estimates, "./debug/lasso_estimates.txt");
    }

  /*if(rank==1 && bootstraps == 0)
      print(lasso_estimates, "./debug/lasso_estimates_1.txt");
    
  if(rank==2 && bootstraps == 0)
      print(lasso_estimates, "./debug/lasso_estimates_2.txt");

  if(rank==3 && bootstraps==0)
      print(lasso_estimates, "./debug/lasso_estimates_3.txt");*/



    if(rank_e == 0)
    {
      VectorXf::Index mspe_min;
      vector<float> mspemin;
      for(int i=0; i<init->n_lambdas/init->n_miniest; i++)
        if(mspe(i) > 0)
          mspemin.push_back(mspe(i));

      VectorXf mspemin_nz(mspemin.size()); 

      for(int i=0; i<mspemin.size(); i++)
          mspemin_nz(i) = mspemin[i];
        
      print(mspemin_nz, "./debug/mspenz.txt");
  
      //float min_coef = mspe.unaryExpr(ptr_fun(not_NaN)).minCoeff( &mspe_min );
      float min_coef = mspemin_nz.minCoeff(&mspe_min);
      int min_id = (int) mspe_min;
      //whichbest(bootstraps) = min_id;
      best_estimates.row(bootstraps) = lasso_estimates.row(min_id);
    }
  }
  if (MPI_COMM_NULL != comm_r) {
    free(Z_mx_e_1);
    free(Z_stacked_1);
    free(bdata_f_1);
    free(Z_mx_e_2);
    free(Z_stacked_2);
    free(bdata_f_2);
  } 

  /*if ( rank == 0 && init->debug ) {
    print(best_estimates, "./debug/best_estimates.txt"); 
    print(whichbest, "./debug/whichbest.txt");
    }*/

  //free memory
  lasso_estimates.resize(0,0);

  VectorXf coef_;
  MatrixXf best_l1_root;
  
  if (MPI_COMM_NULL != L1_roots_comm_e)
  {
    best_l1_root.setZero(init->n_boots_est, p2);  

   // MPI_Gather(best_estimates.data(), best_estimates.rows() * best_estimates.cols(), MPI_FLOAT,
   // best_l2_root.data(), best_estimates.rows() * best_estimates.cols(), MPI_FLOAT, 0, L1_roots_comm_e);

    MPI_Allgather(best_estimates.data(), best_estimates.rows() * best_estimates.cols(), MPI_FLOAT,
    best_l1_root.data(), best_estimates.rows() * best_estimates.cols(), MPI_FLOAT, L1_roots_comm_e);
  }
  
  
  if (MPI_COMM_NULL != L1_roots_comm_e)
    coef_ =  best_estimates.colwise().mean();

  //VectorXf coef_ = median(best_estimates);

  if (rank == 0)
  {
    time_e = MPI_Wtime() - time_s;
    cout << "Total Estimation time: " << time_e << "(s)" << endl;
    cout << "Total Estimation lasso communication time: " << lasso_est_comm << "(s)" << endl;
    cout << "\t *Kronecker product time: " << time7 << "(s)" << endl;
    cout << "\t *Vectorization time: " << time8 << "(s)" << endl;
    cout << "Total Estimation lasso computation time: " << time_las_e - lasso_est_comm << "(s)" << endl;
    //print(scores, "./debug/scores.txt");
    //print(coef_, "./debug/coef_.txt");  

    time_w_s = MPI_Wtime();
  }


  //Prepare data for bagging. 
  //MatrixXf estimates_l1_root, scores_l2_root, scores_l1_root, scores_me(scores.rows(), scores.cols());
  /*MatrixXf scores_l2_root, scores_l1_root, scores_me(scores.rows(), scores.cols());

    MPI_Allreduce(scores.data(), scores_me.data(), scores.rows() * scores.cols(), MPI_FLOAT, MPI_SUM, comm_me); //we can do a reduce too.
    scores_me /= nprocs_me;

    if (MPI_COMM_NULL != L2_roots_comm_e) {
    scores_l2_root.setZero(init->n_boots_est/init->n_est, init->n_lambdas);  

    MPI_Gather(scores_me.data(), scores_me.rows() * scores_me.cols(), MPI_FLOAT,
    scores_l2_root.data(), scores_me.rows() * scores_me.cols(), MPI_FLOAT, 0, L2_roots_comm_e);

    }

    if (MPI_COMM_NULL != L1_roots_comm_e)
    {
    scores_l1_root.setZero(init->n_boots_est, init->n_lambdas);  

    MPI_Allgather(scores_l2_root.data(), scores_l2_root.rows() * scores_l2_root.cols(), MPI_FLOAT,
    scores_l1_root.data(), scores_l2_root.rows() * scores_l2_root.cols(), MPI_FLOAT, L1_roots_comm_e);
    }*/

  //write data into a hdf5 file
  if ( MPI_COMM_NULL != L1_roots_comm_e)
  {

    //VectorXf _coef_(p2);
    //_coef_ = coef_; 
    Map<MatrixXf> _coef_(coef_.data(),p,p);
    float *b_hat;
    float *bic_scores;
    b_hat  = (float*) malloc ( (p2/l1_root_size_e) * sizeof(float) );
    //b_hat = (float *) malloc ((p/l1_root_size_e) * p * sizeof(float));
   // bic_scores = (float* ) malloc (scores_l1_root.rows()/l1_root_size_e *  scores_l1_root.cols() * sizeof(float) );;
    //Map<VectorXf> (b_hat, p2/l1_root_size_e) =
    //  _coef_.segment(l1_root_rank_e * (p2/l1_root_size_e), (p2/l1_root_size_e) );

    Map<MatrixXf> (b_hat, p/l1_root_size_e, p) = 
      _coef_.block(l1_root_rank_e * p/l1_root_size_e,
          0, p/l1_root_size_e, p);


    //Map<MatrixXf> (bic_scores, scores_l1_root.rows()/l1_root_size_e, scores_l1_root.cols()) =
    //  scores_l1_root.block(l1_root_rank_e * scores_l1_root.rows()/l1_root_size_e,
    //      0, scores_l1_root.rows()/l1_root_size_e, scores_l1_root.cols());

    write_out (p, p, b_hat, init->Outfile1, L1_roots_comm_e, "coef_");
    //write_out (init->n_boots_est, init->n_lambdas, bic_scores, init->Outfile2, L1_roots_comm_e, "R2"); 


  }

  if (rank == 0)
  {
    time_w_e = MPI_Wtime() - time_w_s;
    cout << "Save time: " << time_w_e << "(s)" << endl;

  }

  if ( (rank == 0) && init->verbose) { cout << "Final Max score --> " << scores.unaryExpr(ptr_fun(not_NaN)).maxCoeff() << endl; }
  if ( (rank == 0)) { total_end = MPI_Wtime() - total_start; cout << "Final End time: " << total_end << "(s)" << endl;}
  if ( (rank == 0)) {cout << "---> UoI VAR complete." << endl;}

  return coef_;

}


  MatrixXf
lasso_sweep (float *data_f, int local, int p, VectorXf lambda_, int qrows, int N, int L, int D, int boots_sel, bool debug, int MAX_ITER, float RELTOL, float ABSTOL, float rho, MPI_Comm world_comm, MPI_Comm comm_e, MPI_Comm comm_sweep, MPI_Comm comm_r, int n_readers)
{

  int rank_world, nprocs_world;
  MPI_Comm_rank(world_comm, &rank_world);
  MPI_Comm_size(world_comm, &nprocs_world);

  int rank_sweep, nprocs_sweep;
  MPI_Comm_rank(comm_sweep, &rank_sweep);
  MPI_Comm_size(comm_sweep, &nprocs_sweep);

  int rank_r, nprocs_r;
  if(MPI_COMM_NULL != comm_r) {
    MPI_Comm_rank(comm_r, &rank_r);
    MPI_Comm_size(comm_r, &nprocs_r);
  }

  int bootstrap_subsamples; //total number of subsamples closer to total samples no circular dependence
#ifdef CIRCULARDEPENDENCE
  bootstrap_subsamples = N;
#else
  bootstrap_subsamples = N - (N%L); 
#endif

  //get the shape of data matrix.
  int n_samples = local; // samples here are samples per core. 
  int n_features = p;
  int n_lambdas = lambda_.size();

  int size_s = (N-L+2) * L;

  if ( rank_world == 0 && debug) {
    Map<MatrixXf> data_(data_f, local, p);
    print(data_, "./debug/data_0.txt");
  }


  //if(MPI_COMM_NULL != comm_r) {
  //  float *data_f;
  //  data_f = (float*)malloc(n_samples * n_features * sizeof(float) );
  //  Map<MatrixXf> (data_f, data_.rows(), data_.cols() ) = data_;
  // }

  MatrixXf X;  
  VectorXf Y;

  /*sizes for distribution*/

  int z_rows;
  if(MPI_COMM_NULL != comm_r) 
    z_rows = bin_size_1D(rank_r, (bootstrap_subsamples-D)*D, nprocs_r);

  int y_rows = bin_size_1D(rank_sweep, (bootstrap_subsamples-D), nprocs_sweep); //Remove this line if unnecessary.

  int kron_rows = bin_size_1D(rank_sweep, (bootstrap_subsamples-D)*n_features, nprocs_sweep);

    //if (rank_sweep == 0 ) cout << "sel qrows: " << qrows << " zrows: " << z_rows << " y_rows: " << y_rows << endl;

  //Containers to store the estimates and scores.
  MatrixXf estimates(boots_sel * n_lambdas, n_features * n_features );
  VectorXf est_(n_features * n_features);
  MatrixXf B_out, Z_mx; 

  //MatrixXf scores(boots_sel, n_lambdas);
  estimates.setZero();
  //scores.setZero();

  double time1, time2, time3, time4, time5, time6, time7, time8, lasso_comm, lasso_sel_comm;
  float *bdata_f, *Z_mx_e, *Z_stacked; 

  if (MPI_COMM_NULL != comm_r) {
    bdata_f = (float *)malloc(qrows * n_features * sizeof(float) );
    Z_mx_e = (float *)malloc(z_rows * n_features * sizeof(float));
    Z_stacked = (float *)malloc(z_rows * D*n_features * sizeof(float));
  }

  for ( int bootstraps = 0; bootstraps < boots_sel; bootstraps++) 
  {

   // if(rank_sweep==0)
   //   cout << "boostraps: " << bootstraps << endl;

    //distribute the data first
    if (rank_world==0)
      time1 = MPI_Wtime();

    if(MPI_COMM_NULL != comm_r)
      var_distribute_data(data_f, qrows, qrows, N, n_features, size_s, bdata_f, L, D, comm_r, comm_r, n_readers );


   // if(rank_sweep==0)
   //   cout << "var_dis_passed" << endl;

    MPI_Barrier(world_comm); 

    /*if(MPI_COMM_NULL != comm_r) {
      if ( rank_r == 0 && bootstraps == 0 && debug) {
        Map<MatrixXf> bdata_(bdata_f, qrows, n_features);
        print(bdata_, "./debug/bdata_0.txt");
      }
    }*/

    if ( rank_world == 0 && bootstraps == 0 )
    {
      time2 = MPI_Wtime() - time1;
      cout << "var_dis_time: " << time2 << "(s)" << endl;
    }

    /* Generate Z_mx matrix */

    if (rank_sweep == 0)
      time3 = MPI_Wtime();

    //var_generate_Z(bdata_f, qrows, z_rows, N, n_features, Z_mx_e, L, D, world_comm, comm_sweep);
    if(MPI_COMM_NULL != comm_r)
      var_generate_Z(bdata_f, qrows, z_rows, N, n_features, Z_mx_e, L, D, comm_r, comm_r, n_readers);

    //if(rank_sweep==0)
    //  cout << "var_gen_Z_passed" << endl;

    if ( rank_world == 0 ) 
    {
      time4 += MPI_Wtime() - time3;
    
      if(bootstraps == 0)
        cout << "var_gen_Z_time: " << time4 << "(s)" << endl;
    }         
    if(MPI_COMM_NULL != comm_r) {
      Z_mx = stack_Z(Z_mx_e, z_rows, n_features, D);
      //cout << "completed stack" << endl;
      Map<MatrixXf>(Z_stacked, Z_mx.rows(), Z_mx.cols()) = Z_mx; 
    }

    if ( rank_world == 0 && bootstraps == 0 && debug ) {
      Map<MatrixXf> Z_st(Z_stacked, Z_mx.rows(), Z_mx.cols());
      //cout << "completed Z-st map" << endl;
      print(Z_st, "./debug/Z_st_0.txt");
      print(Z_mx, "./debug/Z_mx_0.txt");
      //print(Z_stacked, "./debug/Z_stacked_0.txt");
    }
    //This line has been commented out because it causes memory out-of-range problem when more cores are being used.
    //int kron_rows = y_rows * n_features; 

    if (rank_world == 0) {
      time5 = MPI_Wtime();
      //cout << "Before VAR_KRON" << endl; 
    }

    X = var_kron (Z_stacked, Z_mx.rows(), kron_rows, bootstrap_subsamples, n_features, D, comm_r, world_comm, comm_e, n_readers); 


    //if(rank_sweep==0)
    //  cout << "var_kron_passed" << endl;

    //if(rank_sweep == 0)
    //  cout << "Size after var_kron: " << X.rows() << " nprocs: " << nprocs_sweep << " cols: " << X.cols() << endl;
  
    if(X.rows() == 0)
      cout << "sel rank: " << rank_sweep << " has the selection X as 0. " << endl;

    if( rank_world == 0 && bootstraps == 0 && debug)
      print(X, "./debug/X_0.txt");

    //if( rank_sweep == 1 && bootstraps == 0 && debug)
    //  print(X, "./debug/X_1.txt");

    /*if( rank_sweep == 68 && bootstraps == 0 && debug)
      print(X, "./debug/X_68.txt");*/


    /*MPI_Barrier(world_comm);
      for(int ii=0; ii<kron_rows; ii++)
      if(X.row(ii).isZero() && rank_sweep == 68 && bootstraps == 0) {
      MPI_Barrier(world_comm);
      MPI_Abort(world_comm, 911);
      }*/

    if ( rank_world == 0)
    {
      time6 += MPI_Wtime() - time5;
      
      if(bootstraps == 0)
      cout << "var_kron_time: " << time6 << "(s)" << endl;
    }

    if (rank_sweep == 0) {
      time7 = MPI_Wtime();
      //cout << "B1: " << bootstraps << endl; 
    }  

    MPI_Barrier(world_comm);

    Y = var_vectorize (bdata_f, qrows, kron_rows, bootstrap_subsamples, n_features, D, comm_r,  world_comm, comm_e, n_readers);
    //Y = var_vectorize(data_f, qrows, y_rows, bootstrap_subsamples, n_features, D, L, comm_r, world_comm, comm_sweep);

    //if(rank_sweep==0)
    //  cout << "var_vec_passed" << endl;

    if ( rank_world == 0 && bootstraps == 0 && debug )
      print(Y, "./debug/Y_0.txt");

    //if ( rank_sweep == 1 && bootstraps == 0 && debug )
     // print(Y, "./debug/Y_1.txt");

    //MPI_Barrier(world_comm);
    //MPI_Abort(world_comm, 111);
    // block to find the mean of the predictor using MPI_Gather. 
    /*{
      int *displs = NULL;
      int *sizecounts = NULL;
      int recvsize=0; 

      if (rank_sweep == 0)
      sizecounts = (int*) malloc( nprocs_sweep * sizeof(int));

      int Y_size = Y.size();

      MPI_Gather(&Y_size, 1, MPI_INT, sizecounts, 1, MPI_INT, 0, comm_sweep);

      if (rank_sweep == 0) {
      int totlen = 0;
      displs = (int*) malloc( nprocs_sweep * sizeof(int) );

      displs[0] = 0;
      totlen += sizecounts[0];

      for (int i=1; i<nprocs_sweep; i++) {
      totlen += sizecounts[i];
      displs[i] = displs[i-1] + sizecounts[i-1];
      }

      for (int i=0; i<nprocs_sweep; i++)
      recvsize += sizecounts[i];

      Y_m.setZero(recvsize);
      }

      MPI_Gatherv(Y.data(), Y.size(), MPI_FLOAT, Y_m.data(), sizecounts, displs, MPI_FLOAT, 0, comm_sweep);  

      if (rank_sweep == 0) 
      Y_mean = Y_m.mean(); 

      MPI_Bcast(&Y_mean, 1, MPI_FLOAT, 0, comm_sweep); 
      }*/

    if ( rank_world == 0)
    {
      time8 += MPI_Wtime() - time7;
      
      if(bootstraps == 0)
        cout << "var_vectorize_time: " << time8 << "(s)" << endl;
    }   

    for ( int lambda_idx = 0; lambda_idx < n_lambdas; lambda_idx++)
    {
      //if(rank_sweep==0)
      //  cout << "lamda: " << lambda_idx << endl;
      float n_lamb = lambda_(lambda_idx);

      if ( rank_world == 0 ) {
        time1 = MPI_Wtime(); 
      }

      boost::tie(est_,lasso_comm) = lasso(X, Y, n_lamb, MAX_ITER, RELTOL, ABSTOL, rho, comm_sweep);

      if ( rank_world == 0 && bootstraps == 0 && lambda_idx == 0 && debug ) {
        print(est_, "./debug/est_1.txt");
      }

      if(rank_world==0)
      {
        time2 += MPI_Wtime() - time1;
        lasso_sel_comm += lasso_comm;
      }

      //estimates.row((bootstrap*n_lambdas)+lambda_idx) = est_; this stores estimates from 0-n_lambdas consecutively 
      estimates.row((lambda_idx*boots_sel) + bootstraps) = est_;
    }
  }
  if (MPI_COMM_NULL != comm_r) {
    free(Z_mx_e);
    free(Z_stacked);
    free(bdata_f);
  }

  //free(data_f);
  /*if(MPI_COMM_NULL != comm_r) {
    B_out.resize(0,0);
    Z_stacked.resize(0,0);
    Z_mx.resize(0,0);

    }*/

  if ( rank_world == 0 )
  {
    cout << "Total Selection lasso time: " << time2 << "(s)" << endl;
    cout << "Total Lasso communication time: " << lasso_sel_comm << "(s)" << endl;
    cout << "\t *Kronecker Product time: " << time6 << "(s)" << endl;
    cout << "\t *Vectorization time: " << time8 << "(s)" << endl;
    cout << "Total Lasso computation time: " << time2 << "(s)" << endl;
  }    

  return estimates;   

}

MatrixXf CreateZ (MatrixXf mat, int D) 
{

  MatrixXf Out(mat.rows()-D, D*mat.cols());

  for(int i=0; i<mat.rows()-D; i++) {
    for(int j=0; j<D;j++) {
      Out.row(i).segment((j*mat.cols()), mat.cols()) = mat.row((i+j)+D);
    }
  }

  return Out;

}

void print( MatrixXf m, string name )
{
  std::ofstream file(name);
  if (file.is_open())
  {
    file  << m << '\n';
  }

}

void print_int( VectorXi  m, string name )
{
  std::ofstream file(name);
  if (file.is_open())
  {
    file  << m << '\n';
  }

}

void print_vint( vector<int> m, string name )
{
  std::ofstream file(name);
  if (file.is_open())
  {
    for (int i=0; i<m.size(); i++)
      file  << m[i] << '\n';
  }
}

VectorXf logspace (int start, int end, int size) {

  VectorXf vec;
  vec.setLinSpaced(size, start, end);

  for(int i=0; i<size; i++)
    vec(i) = pow(10,vec(i));

  return vec;
}

float pearson (VectorXf vec1, VectorXf vec2) {


  VectorXd vec1_d = vec1.cast <double>();
  VectorXd vec2_d = vec2.cast <double>();

  gsl_vector_view gsl_x = gsl_vector_view_array( vec1_d.data(), vec1_d.size());
  gsl_vector_view gsl_y = gsl_vector_view_array( vec2_d.data(), vec2_d.size());

  gsl_vector *gsl_v1 =  &gsl_x.vector;
  gsl_vector *gsl_v2 = &gsl_y.vector;
  double r = gsl_stats_correlation (gsl_v1->data, 1, gsl_v2->data, 1, gsl_v1->size);

  float r2 = (float) pow(r, 2);

  return r2;
}

MatrixXf KroneckerProduct( MatrixXf m1, MatrixXf m2, int rank)
{
  MatrixXf m3(m1.rows()*m2.rows(), m1.cols()*m2.cols());

  for (int i = 0; i < m1.rows(); i++) {
    for (int j = 0; j < m1.cols(); j++) {
      m3.block(i*m2.rows(), j*m2.cols(), m2.rows(), m2.cols()) =  m1(i,j)*m2;
    }

  }

  return m3;

}


VectorXf VectorizeY( MatrixXf v1 )
{
  VectorXf v2( v1.rows() * v1.cols() ) ;

  for ( int i=0; i<v1.cols(); i++)
    v2.segment(i * v1.rows(), v1.rows()) = v1.col(i);

  return v2;

}

MatrixXf CreateSupport(int n_lambdas, int n_bootstraps, int threshold_, MatrixXf estimates)
{
  // creates supports for the estimates from model selection:
  // Input:
  //------------------------------------------------
  // n_lambdas    : int number of lambda parameters
  // n_bootstraps : int number of sel bootstraps. 
  // threshold_   : int used for soft thresholding
  //estimates     : (n_lambda) x (n_bootstraps) x (n_features) 

  //Output:
  //------------------------------------------
  // support      : (n_lambda) x (n_features) support 
  //TODO: support matrix is currently floa. Must check compatability and convert it into bool.


  /*
     The estimates matrix is stores in n_lambda order, meaning estimates.row(0) = lasso_est of bootstrap(0) with lambda(0), estimates.row(1) = lasso_est of bootstrap(1) with lambda(0)
     estimates.row(2) = lasso_est of bootstrap(2) with lambda(0) and so on. (Converted 3d matrix from python code to 2d matrix.)

     while creating supports support.row(0) = features with lambda(0) selected across n-sel-bootstraps.

     Explanation of the nested loop:
     for each lambda_index
     for each bootstrap_index
     support(0) = intersect lambda selected across n-sel-bootstraps 




   */
  int n_features = estimates.cols();
  MatrixXf support(n_lambdas, n_features);
  MatrixXi tmp(n_bootstraps-1, n_features);

  for (int lambda_idx = 0; lambda_idx < n_lambdas; lambda_idx++)
  {
    tmp.setZero();
    for (int bootstraps = 0; bootstraps < n_bootstraps-1; bootstraps++)
    {
      for (int feature_idx = 0; feature_idx < n_features; feature_idx++)
      {
        //if (estimates(((n_lambdas*bootstraps)+lambda_idx), feature_idx) != 0)
        if(estimates((lambda_idx*n_bootstraps)+(bootstraps+1), feature_idx) != 0)
          tmp(bootstraps, feature_idx) = 1.0;
        else
          tmp(bootstraps, feature_idx) = 0.0;

      }

    }

    VectorXi sum_v(n_features);

    sum_v = tmp.colwise().sum();

    //print(estimates, "./debug/estimates_here.txt");

    //print_int(sum_v, "./debug/sum_v" + to_string(lambda_idx) + ".txt");    

    for (int l = 0; l < n_features; l++)
    {
      if (sum_v(l) >= threshold_) //if (sum_v(l) !=0)
        support(lambda_idx, l) = 1.0;
      else
        support(lambda_idx, l) = 0.0;

    }

  }

  return support;
}

MatrixXf stack_Z (float* mat_f, int rows, int cols, int D) 
{

  Map<MatrixXf> mat(mat_f, rows, cols);

  //if (rank == 0) {
  // cout << "rows " << rows << "\tcols " << cols << "\tmat.rows " << mat.rows() << "\tmat.cols " << mat.cols() << endl;
  //}


  MatrixXf Out(mat.rows(), D*mat.cols());

  if(D>1) {
    for(int i=0; i<mat.rows(); i++) {
      for(int j=0; j<D;j++) {
        Out.row(i).segment((j*mat.cols()), mat.cols()) = mat.row((i+j)+D-1);
      }
    }
  }
  else
    Out = mat;

  return Out;

}

boost::tuple<MatrixXf, MatrixXf, vector<int> > ApplySupport(MatrixXf& H, MatrixXf& H2, VectorXf support)
{

  vector<int> ids;

  for(int i = 0; i<support.size(); i++)
    if (support(i) !=0 )
      ids.push_back(i);

  MatrixXf ret(H.rows(), ids.size());
  MatrixXf ret2(H2.rows(), ids.size());

  for(int i=0; i<ids.size(); i++) {
    ret.col(i) = H.col(ids[i]);
    ret2.col(i) = H2.col(ids[i]); 
  }

  return boost::make_tuple(ret,ret2,ids);

}

MatrixXf ApplySupport_test(MatrixXf& H, vector<int> ids, int p_2)
{
  MatrixXf ret(H.rows(), p_2);

  //for(int i=0; i<ids.size(); i++)
  //  ret.col(ids[i]) = H.col(i);
  for(int i=0; i<ids.size(); i++)
    ret.block(0, ids[i], H.rows(), 1) = H.col(i);      

  //  for(int i=0; i<ids.size(); i++)
  //    for(int j=0; j<H.rows(); j++)
  //      ret(j,ids[i]) = H(j,i);

  cout << "Passed copy." << endl;
  print(ret, "./debug/ret.txt");
  //ret.col(ids[i]) = H.col(i);

  return ret;

}

VectorXf median (MatrixXf mat) {

  VectorXf get_col;
  vector<float> v;

  for (int i=0; i<mat.cols(); i++) {
    get_col = mat.col(i);
    nth_element (get_col.data(), get_col.data()+ get_col.size()/2,
        get_col.data()+get_col.size());

    v.push_back(get_col(get_col.size()/2));

  }

  Map<VectorXf> vec (v.data(), v.size());

  return vec;
}


float explained_variable_R2(VectorXf y_hat, VectorXf y_obs)
{
  cout << "y_hat.size(): " << y_hat.size() << " y_obs.size(): " << y_obs.size() << endl;
  assert(y_hat.size()==y_obs.size());
  float y_dash = y_obs.mean();
  float SSE_tot = (y_obs.array() - y_dash).array().square().sum();
  float SSE_res = (y_obs - y_hat).array().square().sum();
  float R2 = 1 - (SSE_res/SSE_tot);
  return R2;


}

