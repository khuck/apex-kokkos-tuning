#!/bin/bash -e

# Note: many extra Kokkos CMake variables are set to help with testing code changes.
# A minimal working set of flags would be:
# -DKokkos_ENABLE_TUNING=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON \
# -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_AMPERE80=ON \
# -DKokkos_ARCH_NATIVE=ON

module load cuda/11.7 cmake

dokokkos() {
rm -rf build8
set -x
cmake -B build8 \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_C_COMPILER=gcc \
-DCMAKE_INSTALL_PREFIX=`pwd`/install8 \
-DKokkos_ENABLE_TUNING=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_CUDA=ON \
-DKokkos_ENABLE_CUDA_LAMBDA=ON \
-DKokkos_ARCH_AMPERE80=ON \
-DKokkos_ARCH_NATIVE=ON \
-DKokkos_ENABLE_COMPILER_WARNINGS=ON \
-DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
-DKokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
-DAPEX_WITH_CUDA=TRUE \
-DCUDAToolkit_ROOT=${CUDA} \
.

cmake --build build8 --parallel 16
cmake --build build8 --parallel --target install
export CUDA_VISIBLE_DEVICES=0
cmake --build build8 --target tuning.tests
}

dokokkos

