cmake_minimum_required(VERSION 3.20.1) # Minimum required for CUDA

# set the project name and version
project(occupancy VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(cmake/AddGitSubmodule.cmake)

message(INFO " Checking for Kokkos installation in $Kokkos_ROOT...")
add_git_submodule(kokkos)
# Tell APEX where to find Kokkos include headers
set(Kokkos_ROOT ${PROJECT_SOURCE_DIR}/kokkos)
add_git_submodule(apex)

enable_testing()
add_subdirectory(tests)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "----------------------------------------------------------------------")
message(STATUS "Variable Report:")
message(STATUS "----------------------------------------------------------------------")
dump_cmake_variables("^Kokkos")
dump_cmake_variables("^APEX")
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
MESSAGE(STATUS "Libraries: " ${LIBS})
MESSAGE(STATUS "Compiler cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
MESSAGE(STATUS "Compiler cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
MESSAGE(STATUS "Compiler cxx min size flags:" ${CMAKE_CXX_FLAGS_MINSIZEREL})
MESSAGE(STATUS "Compiler cxx flags:" ${CMAKE_CXX_FLAGS})
MESSAGE(STATUS "Install Prefix:" ${CMAKE_INSTALL_PREFIX})
message(STATUS "----------------------------------------------------------------------")

