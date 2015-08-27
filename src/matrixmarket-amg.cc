#include"config.h"
#include<dune/istl/bcrsmatrix.hh>
#include<dune/istl/bvector.hh>
#include<dune/istl/preconditioners.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/paamg/pinfo.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/matrixmarket.hh>

#include<iostream>
#define BS 1
int main()
{
  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> BVector;
  BCRSMat A;
  BVector b;
  loadMatrixMarket(A, std::string("A.txt"));
  loadMatrixMarket(b, std::string("z.txt"));
  BVector x(b.size());
  x=0.0;

  typedef Dune::MatrixAdapter<BCRSMat,BVector,BVector> Operator;
  typedef Dune::Amg::FirstDiagonal Norm;
  typedef Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<BCRSMat,Norm> >
          Criterion;
  typedef Dune::SeqSSOR<BCRSMat,BVector,BVector> Smoother;
  //typedef Dune::SeqSOR<BCRSMat,BVector,BVector> Smoother;
  //typedef Dune::SeqJac<BCRSMat,BVector,BVector> Smoother;
  //typedef Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::MultiplicativeSchwarzMode> Smoother;
  //typedef Dune::SeqOverlappingSchwarz<BCRSMat,BVector,Dune::SymmetricMultiplicativeSchwarzMode> Smoother;
  //typedef Dune::SeqOverlappingSchwarz<BCRSMat,BVector> Smoother;
  typedef typename Dune::Amg::SmootherTraits<Smoother>::Arguments SmootherArgs;
  SmootherArgs smootherArgs;

  smootherArgs.iterations = 1;

  //smootherArgs.overlap=SmootherArgs::vertex;
  //smootherArgs.overlap=SmootherArgs::none;
  //smootherArgs.overlap=SmootherArgs::aggregate;

  smootherArgs.relaxationFactor = 1;
  int coarsenTarget = 2000;
  Criterion criterion(15,coarsenTarget);
  criterion.setDefaultValuesIsotropic(2);
  criterion.setAlpha(.67);
  criterion.setBeta(1.0e-4);
  criterion.setMaxLevel(15);
  criterion.setSkipIsolated(false);
  Dune::SeqScalarProduct<BVector> sp;
  typedef Dune::Amg::AMG<Operator,BVector,Smoother> AMG;

  Dune::Timer watch;
  watch.reset();
  
  Operator fop(A);
  AMG amg(fop, criterion, smootherArgs, 1, 1, 1, false);


  double buildtime = watch.elapsed();

  std::cout<<"Building hierarchy took "<<buildtime<<" seconds"<<std::endl;

  Dune::GeneralizedPCGSolver<BVector> amgCG(fop,amg,1e-6,80,2);
  //Dune::LoopSolver<BVector> amgCG(fop, amg, 1e-4, 10000, 2);
  watch.reset();
  Dune::InverseOperatorResult r;
  amgCG.apply(x,b,r);

  double solvetime = watch.elapsed();

  std::cout<<"AMG solving took "<<solvetime<<" seconds"<<std::endl;

  std::cout<<"AMG building took "<<(buildtime/r.elapsed*r.iterations)<<" iterations"<<std::endl;
  std::cout<<"AMG building together with solving took "<<buildtime+solvetime<<std::endl;
}
