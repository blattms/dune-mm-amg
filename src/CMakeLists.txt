set(noinst_PROGRAMS  matrixmarket-amg)
add_executable(matrixmarket-amg matrixmarket-amg.cc)
target_link_libraries(matrixmarket-amg ${DUNE_LIBS})
