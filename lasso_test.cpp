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

using namespace std;
using namespace Eigen;

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<float> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor> >(values.data(), rows, values.size()/rows);
}

void print( MatrixXf m, string name )
{
  std::ofstream file(name);
  if (file.is_open())
  {
    file  << m << '\n';
  }

}


MatrixXf readfromfile(const string &path, int nrows, int ncols)
{
MatrixXf X = MatrixXf::Zero(nrows,ncols);
ifstream fin (path);

if (fin.is_open())
{
    for (int row = 0; row < nrows; row++)
        for (int col = 0; col < ncols; col++)
        {
            float item = 0.0;
            fin >> item;
            X(row, col) = item;
        }
    fin.close();
}

return X;

}


int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);
  int rank, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  MatrixXf X;
  X = readfromfile("./debug/X_0.txt", 3880, 1600);//= load_csv<MatrixXf>("./debug/X_0.csv");
  VectorXf Y;
  Y = readfromfile("./debug/Y_0.txt", 3880, 1);//load_csv<MatrixXf>("./debug/Y_0.csv");

  print(X, "./data/X.txt");
  print(Y, "./data/Y.txt"); 

  int max_iter = atoi(argv[1]);
  float reltol = atof(argv[2]);
  float abstol = atof(argv[3]);
  float rho = atof(argv[4]);
  float lamb = atof(argv[5]);
  VectorXf z;
  double lasso_comm;
  //MatrixXf X_las = X.cast<float>();
  //VectorXf Y_las = Y.cast<float>();

  boost::tie(z,lasso_comm) = lasso(X, Y, lamb, max_iter, reltol, abstol, rho, MPI_COMM_WORLD);
  
  //VectorXd z_p = z.cast<double>(); 
  print(z, "./data/z.txt");
  
  return 0;

}
