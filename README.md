# apex-kokkos-tuning
Test repo for apex and kokkos autotuning.

## Configure and Build
The steps to configure and build are the same as configuring/building APEX and Kokkos individually. For examples, see:
* [generic-cuda.sh](generic-cuda.sh) - generic NVIDIA workstation example
* [perlmutter.sh](perlmutter.sh) - configure and build for an NVIDIA-GPU HPC system like [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/).
* [frontier.sh](frontier.sh) - configure and build for an AMD-GPU HPC system like [Frontier](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html).

Take note - the only CMake variable that is _required_ is the `-DKokkos_ENABLE_TUNING=ON` setting, which enables the tuning support in Kokkos.

Another note - this repo uses a git submodule that is a fork/branch from the main Kokkos repository. That fork/branch contains the occupancy tuning support that has not yet been merged into the main repository, see PR [6788](https://github.com/kokkos/kokkos/pull/6788).

For slides with an overview of this repository, see [http://www.nic.uoregon.edu/~khuck/kokkos/2024-Kokkos-Tuning-Tutorial/](http://www.nic.uoregon.edu/~khuck/kokkos/2024-Kokkos-Tuning-Tutorial/). To reproduce the `mdrange_gemm` results in the end of the tutorial, see [tutorial.sh](tutorial.sh).

