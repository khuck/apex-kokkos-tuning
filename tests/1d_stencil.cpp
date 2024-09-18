/**
 * 1D_stencil
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

constexpr int length{32768}; // array length
constexpr int lowerBound{100};
constexpr int upperBound{999};

// helper function for matrix init
void initArray(Kokkos::View<double *, Kokkos::HostSpace>& ar, size_t d1) {
    for(size_t i=0; i<d1; i++){
        ar(i)=(rand() % (upperBound - lowerBound + 1)) + lowerBound;
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
        Kokkos::View<double *, Kokkos::HostSpace> left("left stencil", length);
        /* Initialize the view */
        initArray(left, length);
        /* Create a destination view */
        Kokkos::View<double *, Kokkos::HostSpace> right("right stencil", length);
        /* Copy the initial view */
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, right, left);
        /* Create two view references, a source and a destination */
        auto& source = left;
        auto& dest = right;
        /* Simple 1d, 3-point stencil update - use the average of the left, right and current cells */
        const auto kernel = KOKKOS_LAMBDA(const int x) {
            dest(x) = (source(x-1) + source(x) + source(x+1)) / 3.0;
        };
        Kokkos::Profiling::ScopedRegion region("1d_stencil search loop");
        /* We iterate so that we have enough samples to explore the search space.
         * In a real application, this kernel would get called multiple times over
         * the course of a simulation, and would eventually(?) converge. */
        for (int i = 0 ; i < Impl::max_iterations ; i++) {
            fastest_of( "choose_one", 3, [&]() {
                /* Option 1: serial host space */
                Kokkos::parallel_for("serial heat_transfer",
                    Kokkos::RangePolicy<Kokkos::Serial>(min_index,max_index),
                    kernel);
                }, [&]() {
                /* Option 2: dynamic schedule OpenMP host space */
                Kokkos::parallel_for("openmp dynamic heat_transfer",
                    Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::OpenMP>(min_index,max_index),
                    kernel);
                }, [&]() {
                /* Option 3: static schedule OpenMP host space */
                Kokkos::parallel_for("openmp static heat_transfer",
                    Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Static>, Kokkos::OpenMP>(min_index,max_index),
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
