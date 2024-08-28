/**
 * 2D_annealing
 *
 * Complexity: low
 * Tuning problem:
 *
 * Kokkos is executing a simple 2d stencil annealing (heat transfer) problem.
 *
 * This problem uses an MDRange policy for both instances, and the kernel
 * is the same for both instances. However, there are two Engine instances
 * to choose between: Serial and Static OpenMP.
 *
 * In addition, Kokkos will internally tune the tiling factors for the MDRange,
 * for both the serial and the OpenMP instantiations.
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
        int length = 64;
        /* To keep the kernel simple, we don't update first or last cells */
        int min_index = 1;
        int max_index = length - 1;
        Kokkos::View<double **, Kokkos::HostSpace> stencil("stencil", length, length);
        /* Simple 2d, 9-point stencil update - use the average of the surrounding and current cells */
        const auto kernel = KOKKOS_LAMBDA(const int x, const int y) {
            stencil(x,y) = (stencil(x-1,y-1) + stencil(x,y-1) + stencil(x+1,y-1) +
                            stencil(x-1,y)   + stencil(x,y)   + stencil(x+1,y)   +
                            stencil(x-1,y+1) + stencil(x,y+1) + stencil(x+1,y+1)) / 9.0;
        };
        /* We iterate so that we have enough samples to explore the search space.
         * In a real application, this kernel would get called multiple times over
         * the course of a simulation, and would eventually(?) converge. */
        for (int i = 0 ; i < Impl::max_iterations ; i++) {
            fastest_of( "choose_one", 2, [&]() {
                /* Option 1: serial host space */
                Kokkos::parallel_for("serial 2D heat_transfer",
                    Kokkos::MDRangePolicy<Kokkos::Serial,
                        Kokkos::Rank<2>>({min_index, min_index}, {max_index, max_index}),
                    kernel);
                }, [&]() {
                /* Option 2: OpenMP host space */
                Kokkos::parallel_for("openmp 2D heat_transfer",
                    Kokkos::MDRangePolicy<Kokkos::OpenMP,
                        Kokkos::Rank<2>>({min_index, min_index}, {max_index, max_index}),
                    kernel);
                }
            );
        }
    }
    Kokkos::finalize();
}
