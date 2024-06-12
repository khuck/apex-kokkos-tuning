#!/bin/bash -e


dokokkos() {
rm -rf build
set -x
cmake -B build \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_C_COMPILER=gcc \
-DCMAKE_INSTALL_PREFIX=`pwd`/install \
-DKokkos_ENABLE_TUNING=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_CUDA=ON \
-DKokkos_ENABLE_CUDA_LAMBDA=ON \
-DKokkos_ARCH_AMPERE80=ON \
-DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
-DKokkos_ARCH_NATIVE=ON \
-DKokkos_ENABLE_COMPILER_WARNINGS=ON \
-DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
-DKokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
-DAPEX_WITH_CUDA=TRUE \
-DCUDAToolkit_ROOT=${CUDATOOLKIT_HOME} \
.

cmake --build build --parallel 16
cmake --build build --parallel --target install
cmake --build build --target tuning.tests
}

dokokkos

