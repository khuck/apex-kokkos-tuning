#!/bin/bash -e

module reset
module load cmake
module swap PrgEnv-cray PrgEnv-amd
module swap amd amd/5.7.1
module unload darshan-runtime
export CRAYPE_LINK_TYPE=dynamic

dokokkos() {
rm -rf build
set -x
cmake -B build \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_CXX_COMPILER=amdclang++ \
-DCMAKE_C_COMPILER=amdclang \
-DCMAKE_INSTALL_PREFIX=`pwd`/install \
-DKokkos_ENABLE_TUNING=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_HIP=ON \
-DKokkos_ARCH_VEGA90A=ON \
-DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
-DKokkos_ARCH_NATIVE=ON \
-DKokkos_ENABLE_COMPILER_WARNINGS=ON \
-DKokkos_ENABLE_DEPRECATED_CODE_4=ON \
-DKokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
-DAPEX_WITH_HIP=TRUE \
-DAPEX_WITH_PLUGINS=TRUE \
-DROCM_ROOT=${ROCM_PATH} \
.

cmake --build build --parallel 16
cmake --build build --parallel --target install
cmake --build build --target tuning.tests
}

dokokkos

