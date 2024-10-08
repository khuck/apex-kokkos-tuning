/**
 * 1D_annealing
 *
 * Complexity: low
 * Tuning problem:
 *
 * Kokkos is executing a simple 1d stencil annealing (heat transfer) problem.
 *
 * This problem uses a Range policy for all 3 instances, and the kernel
 * is the same for all 3 instances. However, there are three Engine instances
 * to choose between: Serial, Static OpenMP and Dynamic OpenMP.
 *
 */
#include <tuning_playground.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Kokkos::print_configuration(std::cout, false);
        int length = 1000000;
        int min_index = 0;
        int max_index = length - 1;
        Kokkos::View<double *, Kokkos::HostSpace> stencil("stencil", length);
        const auto kernel = KOKKOS_LAMBDA(const int x) {
            if (x == min_index) {
                stencil(x) = (stencil(x) + stencil(x+1)) / 2.0;
            } else if (x == max_index) {
                stencil(x) = (stencil(x-1) + stencil(x)) / 2.0;
            } else {
                stencil(x) = (stencil(x-1) + stencil(x) + stencil(x+1)) / 3.0;
            }
        };
        Kokkos::Profiling::ScopedRegion region("1d_annealing search loop");
        for (int i = 0 ; i < 50 ; i++) {
            fastest_of( "choose_one", 3, [&]() {
                //std::cout << i << " Doing Serial stencil..." << std::endl;
                Kokkos::parallel_for("serial heat_transfer",
                    Kokkos::RangePolicy<Kokkos::Serial>(0,length),
                    kernel);
                }, [&]() {
                //std::cout << i << " Doing Static OpenMP stencil..." << std::endl;
                Kokkos::parallel_for("openmp dynamic heat_transfer",
                    Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::OpenMP>(0,length),
                    kernel);
                }, [&]() {
                //std::cout << i << " Doing Static OpenMP stencil..." << std::endl;
                Kokkos::parallel_for("openmp static heat_transfer",
                    Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Static>, Kokkos::OpenMP>(0,length),
                    kernel);
                }
            );
        }
    }
    Kokkos::finalize();
}
