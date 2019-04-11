#include <iostream>
#include <mpi.h>
//#define EIGEN_DEFAULT_TO_ROW_MAJOR

#include <eigen3/Eigen/Dense>
#include <boost/tuple/tuple.hpp>
#include "CommandLineOptions.h"
#include "UoI_VAR.h"
#include "structure.h"

using namespace std;
using namespace Eigen;

int main( int argc, char* argv[] )
{

  //Initialize MPI for all the processes
  MPI_Init(&argc, &argv);
  int ranks, procs;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &ranks);

  INIT lasso_init;

  CommandLineOptions opts;
  string version = "02.00.000";
  VectorXf coef_;	

  CommandLineOptions::statusReturn_e temp = opts.parse( argc, argv );
  lasso_init.debug = opts.getDebug(); 
  string InputFile = opts.getInputFile();
  string OutputFile1 = opts.getOutputFile1();
  string OutputFile2 = opts.getOutputFile2();
  string data_matrix = opts.getDatasetMatrix(); 
  string data_vector = opts.getDatasetVector();
  lasso_init.Infile = new char[InputFile.length() + 1];
  lasso_init.Outfile1 = new char [OutputFile1.length() + 1];
  lasso_init.Outfile2 = new char [OutputFile2.length() + 1];
  lasso_init.data_mat = new char [data_matrix.length() + 1];
  lasso_init.data_vec = new char [data_vector.length() + 1];
  strcpy(lasso_init.Infile, InputFile.c_str());
  strcpy(lasso_init.Outfile1, OutputFile1.c_str());
  strcpy(lasso_init.Outfile2, OutputFile2.c_str());
  strcpy(lasso_init.data_mat, data_matrix.c_str());
  strcpy(lasso_init.data_vec, data_vector.c_str());
  lasso_init.verbose =  opts.getVerbose();

  if (ranks == 0)
  {

    lasso_init.n_lambdas = opts.getLambdas();
    lasso_init.selection_thres_frac = opts.getSelectionThreshold();
    lasso_init.train_frac_sel = opts.getTrainSelection();
    lasso_init.train_frac_est = opts.getTrainEstimation();
    lasso_init.train_frac_overall = opts.getTrainOverall();
    lasso_init.n_boots_coarse = opts.getBootsCoarse();
    lasso_init.n_boots_sel = opts.getBootsSel();
    lasso_init.n_boots_est = opts.getBootsEst();
    lasso_init.bagging_options = opts.getBaggingOption();
    lasso_init.n_groups = opts.getnGroups();
    lasso_init.n_minigroups = opts.getnMiniGroups();
    lasso_init.n_est = opts.getnEst();  
    lasso_init.n_miniest = opts.getnMiniEst();
    lasso_init.max_iter = opts.getMAXITER();
    lasso_init.reltol = opts.getRELTOL();
    lasso_init.abstol = opts.getABSTOL();
    lasso_init.rho = opts.getRho();
    lasso_init.L = opts.getL();
    lasso_init.D = opts.getD();
    lasso_init.n_readers = opts.getReader();
  }

  //send all the intialized variables from rank 0  to other processes
  MPI_Bcast(&lasso_init.n_lambdas, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.selection_thres_frac, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.train_frac_sel, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.train_frac_est, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.train_frac_overall, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_boots_coarse, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_boots_sel, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_boots_est, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.bagging_options, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_groups, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_minigroups, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_est, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_miniest, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.reltol, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.abstol, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.rho, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.L, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.D, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_readers, 1, MPI_INT, 0, MPI_COMM_WORLD);

  lasso_init.comm = MPI_COMM_WORLD;

  //If all the input parameters are correct call UoI_Lasso(n_lambdas, selection_thres_frac, train_frac_sel, train_frac_est, train_frac_overall, n_boots_coarse, n_boots_sel ...
  // n_boots_est, use_admm, bagging_options, verbose, debug, MPI_COMM_WORLD);

  //UoI_Lasso(struct lasso_init,  MPI_Comm)

  if(CommandLineOptions::OPTS_SUCCESS == temp)
  {
    coef_ = UoI_VAR(&lasso_init);
  }
  else if(CommandLineOptions::OPTS_VERSION == temp)
  {	
    cout << "Version: " << version << endl;
  }
  else if(CommandLineOptions::OPTS_HELP == temp)
  {
    //no need to do anything here.
  }

  else
  {
    cout << "ERROR -- Input options error" << endl;	
  }

  MPI_Finalize();	

  return 0;
}
