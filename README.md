# apex-kokkos-tuning
Test repo for apex and kokkos autotuning.

## Configure and Build
The steps to configure and build are the same as configuring/building APEX and Kokkos individually. For examples, see:
* [generic-cuda.sh](generic-cuda.sh) - generic NVIDIA workstation example
* [perlmutter.sh](perlmutter.sh) - configure and build for an NVIDIA-GPU HPC system like [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/).
* [frontier.sh](frontier.sh) - configure and build for an AMD-GPU HPC system like [Frontier](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html).

Take note - the only CMake variable that is _required_ is the `-DKokkos_ENABLE_TUNING=ON` setting, which enables the tuning support in Kokkos.
