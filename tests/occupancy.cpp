/**
 * occupancy
 *
 * Complexity: low
 *
 * Requires: GPU architecture (tested with cuda)
 *
 * Tuning problem:
 *
 * Kokkos has a "DesiredOccupancy" struct with which users can
 * determine what occupancy is needed. I added this test as it
 * is one of the simplest tests I can imagine, you're tuning a
 * number between 1 and 100. On a V100, at time of writing we
 * tend to see numbers in the 35-45 range be optimal.
 *
 * This is also used on the Kokkos side to verify that
 * RangePolicy Occupancy tuners (the source of these)
 * are effective
 *
 * Note that this currently involves no features.
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
    using exec_space =  Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;
    using host_space = Kokkos::DefaultHostExecutionSpace;
    using view_type = Kokkos::View<double **, memory_space>;

    tuned_kernel(argc, argv,
        [&](const int total_iters) {
            view_type left("process_this", 1000000, 25);
            return std::make_tuple(left, 20);
        },
        [&](const int x, const int total_iters, view_type A, int R) {
            Kokkos::RangePolicy<> p(0, A.extent(0));
            auto const p_occ = Kokkos::Experimental::prefer(
                p, Kokkos::Experimental::DesiredOccupancy{Kokkos::AUTO});
            const int M = A.extent_int(1);
            Kokkos::parallel_for("Bench", p_occ,
                KOKKOS_LAMBDA(int i) {
                    for (int r = 0; r < R; r++) {
                        double f = 0.;
                        for (int m = 0; m < M; m++) {
                            f += A(i, m);
                            A(i, m) += f;
                        }
                    }
                }
            );
        }
    );
}
