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

bool check_tuning(void) {
    // surely there is a way to get this from Kokkos?
    bool tuning = false;
    bool internal_tuning = Kokkos::tune_internals();
    char * tmp{getenv("APEX_KOKKOS_TUNING")};
    if (tmp != nullptr) {
        std::string tmpstr {tmp};
        if (tmpstr.compare("1") == 0) {
            tuning = true;
        }
    }
    if (tuning && internal_tuning) { std::cout << "Tuning!" << std::endl; return true; }
    std::cout << "Not Tuning!" << std::endl;
    return false;
}

int main(int argc, char *argv[]) {
    using exec_space =  Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;
    using host_space = Kokkos::DefaultHostExecutionSpace;
    using view_type = Kokkos::View<double **, memory_space>;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout, false);
    bool tuning = check_tuning();
    view_type left("process_this", 1000000, 25);
    for (int i = 0 ; i < Impl::max_iterations ; i++) {
        Kokkos::RangePolicy<> p(0, left.extent(0));
        auto const p_occ = Kokkos::Experimental::prefer(
            p, Kokkos::Experimental::DesiredOccupancy{Kokkos::AUTO});
        const int M = left.extent_int(1);
        const auto kernel = KOKKOS_LAMBDA(int i) {
                for (int r = 0; r < 25; r++) {
                    double f = 0.;
                    for (int m = 0; m < M; m++) {
                        f += left(i, m);
                        left(i, m) += f;
                    }
                }
            };
        if (tuning) {
            Kokkos::parallel_for("Bench", p_occ, kernel);
        } else {
            Kokkos::parallel_for("Bench", p, kernel);
        }
    }
  }
  Kokkos::finalize();
}
