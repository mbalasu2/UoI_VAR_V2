#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <boost/tuple/tuple.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include <mpi.h>
#include <cmath>
#define EIGEN_USE_MKL_ALL
#include "lasso.h"
#include "structure.h"
#include "manage-data.h"
#include "UoI_Lasso.h"
#include "bins.h"
#include "distribute-data.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>

using namespace std;
using namespace Eigen;

//########################
//Funcs used in UoI_Lasso
//########################
void print_matrix( MatrixXf , string );
void print_vector( VectorXf , string );
VectorXf logspace (int, int, int);
inline float BIC(float, float, float);
VectorXf median (MatrixXf); 
//inline float r2_score (const VectorXf& , const VectorXf&);
float pearson (VectorXf, VectorXf);
inline float r2_score (const VectorXf& , const VectorXf&);
PermutationMatrix<Dynamic,Dynamic> RandomPermute(int);
MatrixXf CreateSupport(int, int, int, MatrixXf);
boost::tuple<MatrixXf, MatrixXf> lasso_sweep (MatrixXf, VectorXf, VectorXf, float, int, bool, bool, int, float, float, float, MPI_Comm);



//########################
//Funcs definitions
//########################


VectorXf UoI_Lasso(INIT *init)
{
	//Initialize MPI for UoI_Lasso processes the processes
  	int rank, nprocs;
	MPI_Comm world_comm = init->comm;
  	MPI_Comm_size(world_comm, &nprocs);
  	MPI_Comm_rank(world_comm, &rank);

	int n_samples_;
	int n_features_;
	if (rank == 0)
	{
		 //extract model dimensions from design matrix
                n_samples_= get_rows(init->Infile, init->data_mat);
                n_features_ = get_cols(init->Infile, init->data_mat);
	}

	MPI_Bcast(&n_samples_, 1, MPI_INT, 0, world_comm);		
	MPI_Bcast(&n_features_, 1, MPI_INT, 0, world_comm);

	if ( (rank == 0 ) && init->debug) {
		cout << "nprocs: " << nprocs << endl;
		cout << "n_samples: " << n_samples_ << endl;
		cout << "n_features: " << n_features_ << endl;
	}
	//check for processors limits
	if ( (rank == 0 ) && (nprocs > n_samples_) ) 
	{
      		printf("must have nprocs < nrows \n");
      		fflush(stdout);
      		MPI_Abort(world_comm, 3);
    	}

    	if ( ( rank == 0) && ( init->n_groups > nprocs ) ) 
	{
      		printf("must have ngroups < nprocs \n");
      		fflush(stdout);
      		MPI_Abort(world_comm, 4);
    	}

	int local_rows = bin_size_1D(rank, n_samples_, nprocs);
	//MatrixXf X;
	Map<Matrix<float, Dynamic, Dynamic, RowMajor>> X( get_matrix(local_rows, n_features_, n_samples_, world_comm, rank, init->data_mat, init->Infile), local_rows, n_features_);
	//VectorXf y;
	Map<VectorXf> y( get_array(local_rows, n_samples_, world_comm, rank, init->data_vec, init->Infile), local_rows);
	
	if ( (rank == 0 ) && init->verbose)
		cout << "(1) Loaded data.\n" << n_samples_ << " samples with " << n_features_ << " features."	<< endl;

	if ( ( rank == 0 ) && init->debug)	
	{
		print_matrix( X, "./debug/X_1.txt" );
		print_vector( y, "./debug/y_1.txt" );
	}

	//combine the matrix and the vector into 1 matrix for random distribution
	MatrixXf A_(local_rows, (n_features_+1));
	A_ << X,y;
	float *A;
	A = (float*) malloc (A_.rows() * A_.cols() * sizeof(float) );
	Map<Matrix<float, Dynamic, Dynamic, RowMajor> > (A, A_.rows(), A_.cols() ) = A_;

  	if ( (rank == 0) && init->debug)
		print_matrix( A_, "./debug/A.txt");
	
	if ( (rank ==0 ) && init->verbose )
		cout << "Preparing data for random distribution..." << endl;


	//create color with init->n_groups for level 1 and level 2 parallelization
  	int color = bin_coord_1D(rank, nprocs, init->n_groups);
  	MPI_Comm comm_g;
  	MPI_Comm_split(world_comm, color, rank, &comm_g);
 
 	int nprocs_g, rank_g;
  	MPI_Comm_size(comm_g, &nprocs_g);
  	MPI_Comm_rank(comm_g, &rank_g);

  	int qrows = bin_size_1D(rank_g, n_samples_, nprocs_g);
  	//int qrows = floor(krows/nprocs_g);
  	int qcols = n_features_ + 1;
	float *B_;
  	B_ = (float *) malloc( qrows * qcols * sizeof(float) );

 	distribute_data (A,  local_rows, qrows, n_samples_, n_features_, n_samples_, B_, world_comm, comm_g);

	if ( (rank == 0 ) && init->verbose )
                cout << "Random distribution done." << endl;
	
	MPI_Barrier( world_comm );

	Map<MatrixXf> B( B_, qrows, qcols);	

	if ( (rank == 0) && init->debug)
                print_matrix( B, "./debug/B.txt");


	MatrixXf X_(X.rows(), X.cols()); 
	X_ = B.block(0, 0, B.rows(), B.cols()-1);
	VectorXf y_(y.size());
	y_ = B.rightCols(1); 
		
	if ( (rank == 0) && init->debug)
	{
		print_matrix(B, "./debug/B.txt");
                print_matrix( X_, "./debug/X_.txt");
		print_vector( y_, "./debug/y_.txt");
	}
	//perform an initial coarse sweep over the lambda parameters
        //this is to zero-in on the relevant regularization region.
	
	VectorXf lambda_coarse(init->n_lambdas);

	if (rank == 0 )
	{
		if (init->n_lambdas == 1)
			lambda_coarse.setOnes();
		else
			lambda_coarse = logspace(-3, 3, init->n_lambdas); 
	}	

	 MPI_Bcast(lambda_coarse.data(), init->n_lambdas, MPI_FLOAT, 0, world_comm);

	if ( rank == 0 )
		cout << "lambda created for " << lambda_coarse.size() << " size." << endl;
	
	MatrixXf estimates_coarse, scores_coarse; 
	//run the coarse lasso sweep	
	boost::tie (estimates_coarse, scores_coarse)  = lasso_sweep (X_, y_, lambda_coarse, init->train_frac_sel, init->n_boots_coarse,
									init->use_admm, init->debug, init->max_iter, init->reltol, init->abstol,
                                                                                init->rho, comm_g);

	if ( (rank == 0) && init->debug)
        {
                print_matrix( estimates_coarse, "./debug/estimated_coarse.txt");
                print_matrix( scores_coarse, "./debug/scores_coarse.txt");
        }

	//deduce the index which maximizes the explained variance over bootstraps
	VectorXf mean;
	mean = scores_coarse.colwise().mean();
	VectorXf::Index lambda_max_idx;
	
	float max = mean.maxCoeff( &lambda_max_idx );
	
	//obtain the lambda which maximizes the explained variance over bootstraps
	float lambda_max = lambda_coarse(lambda_max_idx);
	
	//in our dense sweep, we'll explore lambda values which encompass a
        //range that's one order of magnitude less than lambda_max itself
	float d_lambda = pow(10, floor(log10(lambda_max)-1));	
	
	if ( (rank == 0) && init->debug)
        {
                print_vector( mean, "./debug/mean_vector.txt");
                print_vector( lambda_coarse, "./debug/lambda_coarse.txt");
		cout << "lambda_max: " << lambda_max << endl;

        }

	 //now that we've narrowed down the regularization parameters,
         //we'll run a dense sweep which begins the model selection module of UoI

	//#######################
        //### Model Selection ###
        // #######################


	if ( (rank == 0) && init->verbose)
		cout << "(2) Beginning model selection. Exploring penalty region centered at " << lambda_max << "." << endl;

	VectorXf lambdas(init->n_lambdas);

        if (rank == 0 )
        {
                if (init->n_lambdas == 1)
                        lambdas << lambda_max;
                else
                        lambdas.setLinSpaced(init->n_lambdas, lambda_max - 5 * d_lambda, lambda_max + 5 * d_lambda);
        }	
		
	MPI_Bcast(lambdas.data(), init->n_lambdas, MPI_FLOAT, 0, world_comm);

	MatrixXf estimates_dense, scores_dense;
	boost::tie (estimates_dense, scores_dense)  = lasso_sweep (X_, y_, lambdas, init->train_frac_sel, init->n_boots_sel,
                                                                        init->use_admm, init->debug, init->max_iter, init->reltol, init->abstol, 
										init->rho, comm_g);

	 if ( (rank == 0) && init->debug)
        {
                print_matrix( estimates_dense, "./debug/estimated_dense.txt");
                print_matrix( scores_dense, "./debug/scores_dense.txt");
        }

	//intersect supports across bootstraps for each lambda value
        //we impose a (potentially) soft intersection

	int threshold = (init->selection_thres_frac * init->n_boots_sel);
	
	//create support matrix storage
	MatrixXf supports_(init->n_lambdas, n_features_);
	
	if ( rank == 0)
		supports_ = CreateSupport(init->n_lambdas, init->n_boots_sel, threshold, estimates_dense); 

	 if ( (rank == 0) && init->debug)
                print_matrix( supports_, "./debug/supports_.txt");
	
	MPI_Bcast(supports_.data(), init->n_lambdas*n_features_, MPI_FLOAT, 0, world_comm);

	//#######################
        //### Model Estimation ###
        // #######################

	//we'll use the supports obtained in the selection module to calculate
        //bagged OLS estimates over bootstraps

	if ( (rank == 0 ) && init->verbose )
                        cout << "(3) Model selection complete. Beginning model estimation, with " << init->n_boots_est << " bootstraps" << endl; 

	//create or overwrite arrays to collect final results
	VectorXf   coef_   =  VectorXf::Zero(n_features_);
	float scores_; 
	//determine how many samples will be used for overall training
	int train_split = round( init->train_frac_overall * qrows);
	//determine how many samples will be used for training within a bootstrap	
	int boot_train_split = round( init->train_frac_est * train_split);

	//set up data arrays
	MatrixXf estimates(init->n_boots_est * init->n_lambdas, n_features_);
	MatrixXf scores(init->n_boots_est, init->n_lambdas);
	MatrixXf best_estimates(init->n_boots_est, n_features_);	
	MatrixXf X_train_, X_test_;
	VectorXf y_train_, y_test_;	 
	//either we plan on using a test set, or we'll use the entire dataset for training
	if ( init->train_frac_overall < 1)
	{
			
		PermutationMatrix<Dynamic,Dynamic> perm_(qrows);
                perm_ = RandomPermute(qrows);
                X_ = perm_ * X_;
                y_ = perm_ * y_;
		X_train_ = X_.topRows(train_split);
		y_train_ = y_.head(train_split);
		X_test_ = X_.bottomRows(qrows - train_split);
		y_test_ = y_.tail(qrows - train_split);
	}	
	else
	{	
		X_train_ = X_;
		y_train_ = y_;
	}
	
	//containers for estimation
	VectorXf y_boot, y_hat_boot, y_true_boot, y_test_boot;
        MatrixXf X_boot, X_boot_test;
	VectorXf z, est_, r;
	//iterate over bootstrap samples
	for (int bootstrap=0; bootstrap < init->n_boots_est; bootstrap++)
	{	
		PermutationMatrix<Dynamic,Dynamic> perm_(X_train_.rows());
                perm_ = RandomPermute(X_train_.rows());
                X_train_ = perm_ * X_train_;
                y_train_ = perm_ * y_train_;
		y_boot = y_train_.head(boot_train_split);
		X_boot = X_train_.topRows(boot_train_split);
		X_boot_test = X_train_.bottomRows(train_split-boot_train_split);
		//extract the bootstrap indices, keeping a fraction of the data available for testing
		
		for (int lamb_idx=0; lamb_idx < init->n_lambdas; lamb_idx++)
		{
                        z = lasso(X_boot, y_boot.array()-y_boot.mean(), 0, init->max_iter, init->reltol, init->abstol, init->rho, comm_g);

			//apply support : can be changed if necessary. multiplies elementwise with the supports_ array to select only the supports.
			r = supports_.row(lamb_idx);
			est_ = z.cwiseProduct(r);
			//store the fitted coefficients
			estimates.row((bootstrap*init->n_lambdas)+lamb_idx) = est_;
                        y_hat_boot = X_boot_test * est_;
                        y_test_boot = y_train_.tail(train_split-boot_train_split);
                        y_true_boot = y_test_boot.array()-y_test_boot.mean();
                        //scores(bootstrap, lamb_idx) = r2_score(y_true, y_hat);
			//calculate sum of squared residuals
			float rss = (y_hat_boot.array() - y_true_boot.array()).square().sum();
			//calculate BIC as our scoring function
			scores(bootstrap, lamb_idx) = BIC(n_features_, boot_train_split, rss);
		}
	}

	 if ( (rank == 0) && init->debug)
                print_matrix( scores, "./debug/scores.txt");
	
	switch (init->bagging_options) 
	{
		case 1:
		{
			//bagging option 1: for each bootstrap sample, find the regularization parameter that gave the best results
			for (int bootstrap=0; bootstrap < init->n_boots_est; bootstrap++)
			{
				VectorXf::Index lambda_max;
        			float max = scores.row(bootstrap).maxCoeff( &lambda_max);
				int lambda_max_idx_ = (int) lambda_max; 
				best_estimates.row(bootstrap) = estimates.row((bootstrap*init->n_boots_est)+lambda_max_idx_); 
			}	
		
			//take the median across estimates for the final, bagged estimate
			coef_ = median(best_estimates); 	
			break;
		}
		case 2:
		{
			//bagging option 2: average estimates across bootstraps, and then find the regularization parameter that gives the best results
			VectorXf mean_scores; 
			mean_scores = scores.colwise().mean(); 
			VectorXf::Index lambda_max;
			float max = mean_scores.maxCoeff( &lambda_max);
			int lambda_max_idx_ = (int) lambda_max;

			for (int bootstrap=0; bootstrap < init->n_boots_est; bootstrap++)
				best_estimates.row(bootstrap) = estimates.row(lambda_max_idx_);

			coef_ = median(best_estimates); 
			break;
		}
		default:
		{
			cerr << "Bagging option " << init->bagging_options << " is not available."; 
			break;
		}
	}
	
	if ( init->train_frac_overall < 1)
	{
		//finally, see how the bagged estimates perform on the test set
		VectorXf y_hat_;
                y_hat_ = X_test_ * coef_;
                VectorXf y_true_;
                y_true_ = y_test_.array()-y_test_.mean();
                scores_ = pearson(y_hat_, y_true_);		
	
	}		
	else
		scores_ = 0.0;

	if ( (rank == 0) && init->verbose) { cout << "Final score --> " << scores_ << endl; }		
	if ( (rank == 0) && init->verbose) {cout << "---> UoI Lasso complete." << endl;}

	if ( (rank == 0) && init->debug)
                print_vector( coef_, "./debug/coef_.txt");

	
	/*write data into a hdf5 file*/	
	VectorXf _coef_(n_features_);
	_coef_ = coef_;
	float *b_hat;
	float *bic_scores;
	b_hat  = (float*) malloc ( (n_features_)/nprocs * sizeof(float) );
	bic_scores = (float* ) malloc (scores.rows()/nprocs *  scores.cols() * sizeof(float) );
	Map<VectorXf> (b_hat, (n_features_)/nprocs) = _coef_.segment(rank * (n_features_)/nprocs, (n_features_)/nprocs );
	Map<MatrixXf> (bic_scores, scores.rows()/nprocs, scores.cols()) = 
							scores.block(rank * scores.rows()/nprocs, 0, scores.rows()/nprocs, scores.cols());;

	write_out (n_features_, 1, b_hat, init->Outfile1, world_comm, "coef_");	
	write_out (init->n_boots_est, init->n_lambdas, bic_scores, init->Outfile2, world_comm, "scores_");	
	
	return coef_; 

}

VectorXf logspace (int start, int end, int size) {

    VectorXf vec;
    vec.setLinSpaced(size, start, end);

    for(int i=0; i<size; i++)
        vec(i) = pow(10,vec(i));

    return vec;
}


boost::tuple<MatrixXf, MatrixXf> 
lasso_sweep (MatrixXf X, VectorXf y, VectorXf lambda, float train_frac, int n_bootstraps, bool use_admm, bool debug, int MAX_ITER, float RELTOL, float ABSTOL, float rho, MPI_Comm comm_sweep)
{
	/*
	Perform Lasso regression across bootstraps of a dataset for a sweep
                of L1 penalty values.

                Parameters
                ----------
                X : np.array
                        data array containing regressors; assumed to be 2-d array with
                        shape n_samples x n_features

                y : np.array
                        data array containing dependent variable; assumed to be a 1-d array
                        with length n_samples

                lambdas : np.array
                        the set of regularization parameters to run boostraps over

                train_frac : float
                        float between 0 and 1; the fraction of data to use for training

                n_bootstraps : int
                        the number of bootstraps to obtain from the dataset; each bootstrap
                        will undergo a Lasso regression

                n_minibatch : int
                        number of minibatches to use in case SGD is used for the regression

                use_admm: bool
                        switch to use the alternating direction method of multipliers (
                        ADMM) algorithm

                Returns
                -------
                estimates : np.array
                        predicted regressors for each bootstrap and lambda value; shape is
                        (n_bootstraps, n_lambdas, n_features)

                scores : np.array
                        scores by the model for each bootstrap and lambda
                        value; shape is (n_bootstraps, n_lambdas)
	*/
	int rank_sweep;
	MPI_Comm_rank(comm_sweep, &rank_sweep);

	//get the shape of X matrix.
	int n_samples = X.rows(); // samples here are samples per core.	
	int n_features = X.cols();
	int n_lambdas = lambda.size(); 

	//Containers to store the estimates and scores.
	MatrixXf estimates(n_bootstraps * n_lambdas, n_features);
	MatrixXf scores(n_bootstraps, n_lambdas);
	estimates.setZero();
	scores.setZero();

	
	int n_train_samples = round(train_frac * n_samples);

	/*if ( (rank_sweep == 0) && debug ) 
	{
		cout << " Data ready for UoI " << endl; 
		cout << "n_samples: " << n_samples << " n_train_samples: " << n_train_samples << endl;
	}*/

	//Intermediate containers required for the computation.
	PermutationMatrix<Dynamic,Dynamic> perm(n_samples);
	MatrixXf X_perm(n_samples, n_features);
        VectorXf y_perm(n_samples);
	VectorXf y_train(n_train_samples);
	MatrixXf X_train(n_train_samples, n_features);
	VectorXf est_(n_features);
	VectorXf y_hat(n_samples-n_train_samples);
	VectorXf y_true(n_samples-n_train_samples);
        VectorXf y_test(n_samples-n_train_samples);
	MatrixXf X_test(n_samples-n_train_samples, n_features);


	for (int bootstrap=0; bootstrap < n_bootstraps; bootstrap++)
	{
		perm = RandomPermute(n_samples);

		//Random Shuffle X and y
		X = perm * X;	
		y = perm * y;
	
		//Split X and y into train and test dataset.
		y_train = y.head(n_train_samples);
                X_train = X.topRows(n_train_samples);
		X_test = X.bottomRows(n_samples-n_train_samples);
		y_test = y.tail(n_samples-n_train_samples);
	
		if (rank_sweep == 0 && bootstrap == 0 && debug)
		{	
		 	print_matrix( X_train, "./debug/X_train.txt");	
			print_matrix( X_test, "./debug/X_test.txt");
			print_vector( y_train, "./debug/y_train.txt");
			print_vector( y_test, "./debug/y_test.txt");

		}
		
		for (int lambda_idx=0; lambda_idx < n_lambdas; lambda_idx++)
		{
			//if (rank_sweep==0)
			//	cout << "top lambda_idx: " << lambda_idx << endl;
			float n_lamb = lambda(lambda_idx);
			est_ = lasso(X_train, y_train.array()-y_train.mean(), n_lamb, MAX_ITER, RELTOL, ABSTOL, rho, comm_sweep);

			//estimates.row((bootstrap*n_lambdas)+lambda_idx) = est_; this stores estimates from 0-n_lambdas consecutively 
			estimates.row((lambda_idx*n_bootstraps) + bootstrap) = est_; // this stores estimates in order 0 in n_lambda steps. so bootstraps are strored consecutively.
			y_hat = X_test * est_;
			y_true = y_test.array()-y_test.mean();
			scores(bootstrap, lambda_idx) = pearson(y_hat, y_true);
		
			if (rank_sweep == 0 && bootstrap == 0 && lambda_idx == 0 && debug)
			{
				print_vector( est_, "./debug/est_dense_0_0.txt");
				print_vector( y_hat, "./debug/y_hat_0_0.txt");
				print_vector( y_true, "./debug/y_true_0_0.txt");
				print_matrix( scores, "./debug/scores_0_0.txt");
			}
		}
	}
	
	return boost::make_tuple(estimates, scores);	

}

inline float r2_score (const VectorXf& x, const VectorXf& y)
{
	/*calculates the person R2 values*/
  	const float num_observations = static_cast<float>(x.size());
  	float x_stddev = sqrt((x.array()-x.mean()).square().sum()/(num_observations-1));
  	float y_stddev = sqrt((y.array()-y.mean()).square().sum()/(num_observations-1));
  	float numerator = ((x.array() - x.mean() ) * (y.array() - y.mean())).sum() ;
  	float denomerator = (num_observations-1)*(x_stddev * y_stddev);
	float r2 = pow((numerator / denomerator),2); 
  	return r2;
}

PermutationMatrix<Dynamic,Dynamic> RandomPermute(int rows)
{
	srand(time(0));
	PermutationMatrix<Dynamic,Dynamic> perm_(rows);
        perm_.setIdentity();
        random_shuffle(perm_.indices().data(), perm_.indices().data()+perm_.indices().size());
	
	return perm_;
}

MatrixXf CreateSupport(int n_lambdas, int n_bootstraps, int threshold_, MatrixXf estimates)
{
	// creates supports for the estimates from model selection:
	// Input:
	//------------------------------------------------
	// n_lambdas 	: int number of lambda parameters
	// n_bootstraps	: int number of sel bootstraps. 
	// threshold_	: int used for soft thresholding
	//estimates 	: (n_lambda) x (n_bootstraps) x (n_features) 
	
	//Output:
	//------------------------------------------
	// support	: (n_lambda) x (n_features) support 
	//TODO: support matrix is currently floa. Must check compatability and convert it into bool.



	int n_features = estimates.cols(); 
	MatrixXf support(n_lambdas, n_features);
	MatrixXi tmp(n_bootstraps, n_features);

	for (int lambda_idx = 0; lambda_idx < n_lambdas; lambda_idx++)
	{
		tmp.setZero(); 
		for (int bootstraps = 0; bootstraps < n_bootstraps; bootstraps++)
		{
			for (int feature_idx = 0; feature_idx < n_features; feature_idx++)
			{
				if (estimates(((n_lambdas*bootstraps)+lambda_idx), feature_idx) != 0)
					tmp(bootstraps, feature_idx) = 1.0;
				else
					tmp(bootstraps, feature_idx) = 0.0;
					
			}

		}
	
		VectorXi sum_v(n_features);
		
		sum_v = tmp.colwise().sum();
		
		for (int l = 0; l < n_features; l++)
		{
			if (sum_v(l) >= threshold_)
				support(lambda_idx, l) = 1.0;
			else
				support(lambda_idx, l) = 0.0;

		}

	}

	return support; 
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

inline float BIC(float n_features, float n_samples, float rss)
{
        /*
        Calculate the Bayesian Information Criterion under the assumption of
        normally distributed disturbances (which allows the BIC to take on the
        simple form below).

        Parameters
        ----------
        n_features : int
                number of model features

        n_samples : int
                number of samples in the dataset

        rss : float
                the residual sum of squares

        Returns
        -------
        BIC : float
                Bayesian Information Criterion
        */

        float bic;
        return (bic = -n_samples * log(rss/n_samples) - n_features * log(n_samples));


}

void print_matrix( MatrixXf m, string name )
{
        std::ofstream file(name);
        if (file.is_open())
        {
                file  << m << '\n';
        }

}

void print_vector( VectorXf m, string name )
{
        std::ofstream file(name);
        if (file.is_open())
        {
                file  << m << '\n';
        }

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
