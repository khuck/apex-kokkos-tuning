/**
 * 1D_stencil
 *
 * Complexity: low
 * Tuning problem:
 *
 * Kokkos is executing a simple 1d stencil annealing (heat transfer) problem.
 *
 * This problem uses a Team policy for two instances, and the kernel
 * is the same for both instances. However, there are two Engine instances
 * to choose between: Static OpenMP and Dynamic OpenMP. This example will
 * tune the thread count and chunk size (vector length) too, when executed
 * with the --kokkos-tune-internals flag.
 *
 */
#include <tuning_playground.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>

constexpr int length{1048576}; // array length
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
        const auto kernel = KOKKOS_LAMBDA(const Kokkos::TeamPolicy<Kokkos::OpenMP>::member_type &team_member) {
            // Calculate a global thread id
            int x = team_member.league_rank () * team_member.team_size () +
                team_member.team_rank ();
            if (x < min_index || x == max_index) {
                return;
            }
            dest(x) = (source(x-1) + source(x) + source(x+1)) / 3.0;
        };
        Kokkos::Profiling::ScopedRegion region("1d_stencil_team_auto search loop");
        /* We iterate so that we have enough samples to explore the search space.
         * In a real application, this kernel would get called multiple times over
         * the course of a simulation, and would eventually(?) converge. */
        for (int i = 0 ; i < Impl::max_iterations ; i++) {
            fastest_of( "choose_one", 2, [&]() {
                /* Option 1: dynamic schedule OpenMP host space */
                Kokkos::parallel_for("openmp dynamic heat_transfer",
                    Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::OpenMP>(1,Kokkos::AUTO,Kokkos::AUTO),
                    kernel);
                }, [&]() {
                /* Option 2: static schedule OpenMP host space */
                Kokkos::parallel_for("openmp static heat_transfer",
                    Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Static>, Kokkos::OpenMP>(1,Kokkos::AUTO,Kokkos::AUTO),
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
