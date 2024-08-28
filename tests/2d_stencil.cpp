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

constexpr int length{64};
constexpr int lowerBound{100};
constexpr int upperBound{999};

// helper function for matrix init
void initArray(Kokkos::View<double **, Kokkos::HostSpace>& ar, size_t d1, size_t d2) {
    for(size_t i=0; i<d1; i++){
        for(size_t j=0; j<d2; j++){
            ar(i,j)=(rand() % (upperBound - lowerBound + 1)) + lowerBound;
        }
    }
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Kokkos::print_configuration(std::cout, false);
        /* To keep the kernel simple, we don't update first or last cells */
        int min_index = 1;
        int max_index = length - 1;
        /* Create initial view */
        Kokkos::View<double **, Kokkos::HostSpace> left("left stencil", length, length);
        /* Initialize the view */
        initArray(left, length, length);
        /* Create a destination view */
        Kokkos::View<double **, Kokkos::HostSpace> right("right stencil", length, length);
        /* Copy the initial view */
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, right, left);
        /* Create two view references, a source and a destination */
        auto& source = left;
        auto& dest = right;
        /* Simple 2d, 9-point stencil update - use the average of the surrounding and current cells */
        const auto kernel = KOKKOS_LAMBDA(const int x, const int y) {
            dest(x,y) = (source(x-1,y-1) + source(x,y-1) + source(x+1,y-1) +
                            source(x-1,y)   + source(x,y)   + source(x+1,y)   +
                            source(x-1,y+1) + source(x,y+1) + source(x+1,y+1)) / 9.0;
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
            /* Swap the views */
            auto& tmp = source;
            source = dest;
            dest = tmp;
        }
    }
    Kokkos::finalize();
}
