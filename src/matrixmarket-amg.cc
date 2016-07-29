#include"config.h"
#include<dune/istl/bcrsmatrix.hh>
#include<dune/istl/bvector.hh>
#include<dune/istl/preconditioners.hh>
#include<dune/istl/paamg/amg.hh>
#include<dune/istl/paamg/pinfo.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/matrixmarket.hh>
#include<dune/istl/matrixredistribute.hh>
#include<dune/istl/paamg/graph.hh>
#include<dune/common/parallel/mpihelper.hh>
#include<iostream>

class MPIError {
public:
  /** @brief Constructor. */
  MPIError(std::string s, int e) : errorstring(s), errorcode(e){}
  /** @brief The error string. */
  std::string errorstring;
  /** @brief The mpi error code. */
  int errorcode;
};

void MPI_err_handler(MPI_Comm *comm, int *err_code, ...){
  DUNE_UNUSED_PARAMETER(comm);
  char *err_string=new char[MPI_MAX_ERROR_STRING];
  int err_length;
  MPI_Error_string(*err_code, err_string, &err_length);
  std::string s(err_string, err_length);
  std::cerr << "An MPI Error ocurred:"<<std::endl<<s<<std::endl;
  delete[] err_string;
  throw MPIError(s, *err_code);
}

int main(int argc, char** argv)
{
#if HAVE_PARMETIS
  Dune::MPIHelper::instance(argc, argv);
  MPI_Errhandler handler;
  MPI_Errhandler_create(MPI_err_handler, &handler);
  MPI_Errhandler_set(MPI_COMM_WORLD, handler);
  auto world_comm = Dune::MPIHelper::getCollectiveCommunication();
  const int BS=1; // block size, sparse scalar matrix
  typedef Dune::FieldMatrix<double,BS,BS> MatrixBlock;
  typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
  typedef Dune::FieldVector<double,BS> VectorBlock;
  typedef Dune::BlockVector<VectorBlock> BVector;
  BCRSMat A;
  if(world_comm.rank()==0)
  {
      loadMatrixMarket(A, std::string("A.txt"));
      //loadMatrixMarket(b, std::string("z.txt"));
  }
  typedef std::size_t GlobalId; // The type for the global index
  typedef Dune::OwnerOverlapCopyCommunication<GlobalId> Communication;

  Communication comm(world_comm);
  Communication* comm_redist;
  // No need to add any indices to comm.indexSet()
  BCRSMat parallel_A;
  typedef Dune::Amg::MatrixGraph<BCRSMat> MatrixGraph;
  Dune::RedistributeInformation<Communication> rinfo;
  bool hasDofs = Dune::graphRepartition(MatrixGraph(A), comm,
                                         static_cast<int>(world_comm.size()),
                                         comm_redist,
                                         rinfo.getInterface(),
                                         true);
  rinfo.setSetup();
  redistributeMatrix(const_cast<BCRSMat&>(A), parallel_A, comm, *comm_redist, rinfo);
  BVector b(A.N());
  BVector parallel_b(parallel_A.N());
  BVector parallel_x(parallel_A.M());
  b=100;
  rinfo.redistribute(b, parallel_b);

  if(hasDofs)
  {
    // the index set has changed. Rebuild the remote information
    comm_redist->remoteIndices().rebuild<false>();
    typedef Dune::SeqSSOR<BCRSMat,BVector,BVector> Prec;
    typedef Dune::BlockPreconditioner<BVector,BVector,
                                      Communication,Prec>
        ParPrec; // type of parallel preconditioner
    typedef Dune::OverlappingSchwarzScalarProduct<BVector,Communication>
        ScalarProduct; // type of parallel scalar product
    typedef Dune::OverlappingSchwarzOperator<BCRSMat,BVector,
                                             BVector,Communication>
        Operator; // type of parallel linear operator

    ScalarProduct sp(*comm_redist);
    Operator op(parallel_A, *comm_redist);
    Prec prec(parallel_A, 1, 1.0);
    ParPrec pprec(prec, *comm_redist);
    Dune::InverseOperatorResult r;
    Dune::CGSolver<BVector> cg(op, sp, pprec, 10e-8, 8,
                               world_comm.rank()==0?2:0);
    cg.apply(parallel_x, parallel_b, r);
  }

  BVector x;
  //if(comm_world.rank()==0)
  x.resize(A.M());
  rinfo.redistributeBackward(x, parallel_x);

    /*
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
  typedef Dune::Amg::SmootherTraits<Smoother>::Arguments SmootherArgs;
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
    */
#else
  std::cout<<"This program is only meaningfull with ParMETIS or PT-Scotch"<<std::endl;
#endif
}
