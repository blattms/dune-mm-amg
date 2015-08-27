# set up project
project("dune-mm-amg" C CXX)

# general stuff
cmake_minimum_required(VERSION 2.8.6)

#find dune-common and set the module path
find_package(dune-common)
list(APPEND CMAKE_MODULE_PATH ${dune-common_MODULE_PATH}
     "${CMAKE_SOURCE_DIR}/cmake/modules")# make sure our own modules are found

#include the dune macros
include(DuneMacros)

# start a dune project with information from dune.module
dune_project()

# $Id: duneproject 6400 2011-03-18 16:29:45Z mblatt $
# we need the module file to be able to build via dunecontrol
set(EXTRA_DIST dune.module)
set(SUBDIRS  src doc)
add_subdirectory(cmake/modules)
add_subdirectory(src)
add_subdirectory(doc)
# finalize the dune project, e.g., generate config.h etc.
finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
