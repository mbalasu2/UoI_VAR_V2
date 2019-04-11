# Distributed-UoI_NMF 
			 Updated with BOOST CommandLine and rectified lasso_admm
			 for sparse matrices.
       tested and validated with Matlab results
                    Runs with tested on Cori-KNL NERSC supercomputer.


## Requirements

languages: C, C++

computing resources: The model is created for `Cori KNL`. 

API and Libraries: MPI, HDF5-parallel, eigen3, boost


## Modules and file structure

1. UoI_VAR.cpp contains the UoI_VAR module called from main.cpp.
2. lasso_admm for sparse is the only external library used. 
3. Data read and write modules are coded in manage-data.c
4. Data random distribution module is handled by var-distribute-data.c module.
5. Since the data for var models are time series, the data distribution requires creation of blocks ,
   so a block randomization of the problem is implemented. 
6. A distributed vectorization module is create in var_vectorize.cpp file.
7. A distributed kronecker product is in var_kron.cpp file.

## Data input
1. The input should be in h5 file format.
2. The time series data in h5 file should be in reverse time order, meaning row 1 should constitute time T,
   and row 2 should be time T-1 so on. 
3. The dataset should have the rows as samples and columns as features (n_samples X n_features). 

## TODO

1. Nothing at this points.

#past completed tasks

1. Check the output with the Matlab output
2. Contact Trevor Ruiz regarding the Predict function in Matlab linear model. 
3. Tune Lasso-ADMM hyper parameter to match the output of Matlab and c++ code.
4. This will not change the runtime by much. 

## Installation

1. Clone the module into your directory
2. `source load.sh`
3. `make`

## Usage

1. Edit the "job" script for input file path (vi job)
2. sbatch job (to submit the job)

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

TODO: Write history

## Credits

Author: Mahesh Balasubramanian (guidance from Trevor Ruiz, Kris Bouchard, Prabhat, Brandon Cook)

Version: 2.0


## License

TODO: Write license

## Detailed description of the directory

1. load.sh : Has the required modules for the correct execution of UoI_Lasso application

