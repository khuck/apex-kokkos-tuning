#!/bin/bash -e

# Note: many extra Kokkos CMake variables are set to help with testing code changes.
# A minimal working set of flags would be:
# -DKokkos_ENABLE_TUNING=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON \
# -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_VEGA90A=ON -DKokkos_ARCH_NATIVE=ON

#module reset
#module load cmake
#module swap PrgEnv-cray PrgEnv-amd
#module swap amd amd/5.7.1
#module load rocm/5.7.1
#module unload darshan-runtime
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
-DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
-DKokkos_ENABLE_DEBUG=ON \
-DKokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK=ON \
-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON \
-DKokkos_ARCH_NATIVE=ON \
-DAPEX_WITH_HIP=TRUE \
-DROCM_ROOT=${ROCM_PATH} \
.

cmake --build build --parallel 16
cmake --build build --parallel --target install
cmake --build build --target tuning.tests
}

dokokkos

#-DKokkos_ARCH_VEGA90A=ON \
#-DROCTRACER_ROOT=${ROCM_PATH} \
#-DROCPROFILER_ROOT=${ROCM_PATH} \
#-DROCTX_ROOT=${ROCM_PATH} \
#-DRSMI_ROOT=${ROCM_PATH} \
