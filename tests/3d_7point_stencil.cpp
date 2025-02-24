/**
 * 3D_stencil
 *
 * Complexity: low
 * Tuning problem:
 *
 * Kokkos is executing a simple 3d stencil annealing (heat transfer) problem.
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
#include <Kokkos_Random.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>

constexpr int length{128};

// helper function for matrix init
void initArray(Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space>& ar, size_t d1, size_t d2, size_t d3) {
    const auto kernel = KOKKOS_LAMBDA(const int x, const int y, const int z) {
        ar(x,y,z)= x + y + z;
    };
    Kokkos::parallel_for("initialize",
        Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                            Kokkos::Rank<3>>
            ({0, 0, 0}, {d1, d2, d3}), kernel);
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        Kokkos::print_configuration(std::cout, false);
        /* To keep the kernel simple, we don't update first or last cells */
        int min_index = 1;
        int max_index = length - 1;
        /* Create initial view */
        Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space> left("left stencil", length, length, length);
        /* Initialize the view */
        std::cout << "init..." << std::endl;
        std::cout.flush();
        initArray(left, length, length, length);
        /* Create a destination view */
        Kokkos::View<double ***, Kokkos::DefaultExecutionSpace::memory_space> right("right stencil", length, length, length);
        /* Copy the initial view */
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, right, left);
        /* Create two view references, a source and a destination */
        auto& source = left;
        auto& dest = right;
        /* Simple 3d, 27-point stencil update -
         * use the average of the surrounding and current cells,
         * but don't use diagonals. */
        const auto kernel = KOKKOS_LAMBDA(const int x, const int y, const int z) {
            dest(x,y,z) = (source(x,y,z-1) + source(x,y,z+1) +
                         source(x,y-1,z) + source(x,y,z) + source(x,y+1,z) +
                         source(x-1,y,z) + source(x+1,y,z)) / 7.0;
        };
        std::cout << "compute..." << std::endl;
        std::cout.flush();
        Kokkos::Profiling::ScopedRegion region("3d_stencil search loop");
        /* We iterate so that we have enough samples to explore the search space.
         * In a real application, this kernel would get called multiple times over
         * the course of a simulation, and would eventually(?) converge. */
        for (int i = 0 ; i < 4 * Impl::max_iterations ; i++) {
                Kokkos::parallel_for("3D 7-point jacobi",
                    Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                        Kokkos::Rank<3>>
                        ({min_index, min_index, min_index},
                            {max_index, max_index, max_index}),
                    kernel);
            /* Swap the views */
            auto& tmp = source;
            source = dest;
            dest = tmp;
        }
    }
    Kokkos::finalize();
}
