set(noinst_PROGRAMS  matrixmarket-amg)
add_executable(matrixmarket-amg matrixmarket-amg.cc)
add_dune_superlu_flags(matrixmarket-amg)
add_dune_umfpack_flags(matrixmarket-amg)
target_link_libraries(matrixmarket-amg ${DUNE_LIBS})
