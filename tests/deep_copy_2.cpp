/**
 * Deep Copy
 *
 * Complexity: simple
 * Tuning problem:
 *
 * Kokkos transforms data layouts of View depending on the architecture.
 *
 * That is, in a 3D view, we change which dimension is stride 1 access.
 * This means that in some cases, we need to transpose data if it's
 * operated on in multiple ExecutionSpaces
 *
 * It does so using an "MDRangePolicy," a set of tightly nested loops.
 *
 * These "MDRangePolicy's" have tile sizes, which you're picking.
 * Currently no features, but plans are to vary the size of these Views
 * to enable you to see whether optimal tile sizes vary with View shapes
 *
 * This is basically a smoke-test, can your tool tune tile sizes
 */
#include <tuning_playground.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
int main(int argc, char *argv[]) {
  constexpr const int data_size = 100;
  using left_type = Kokkos::View<float **, Kokkos::LayoutLeft,
                                 Kokkos::DefaultExecutionSpace::memory_space>;
  using right_type = Kokkos::View<float **, Kokkos::LayoutRight,
                                  Kokkos::DefaultExecutionSpace::memory_space>;
  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout, false);
    left_type left("left", data_size, data_size);
    right_type right("right", data_size, data_size);
    for (int i = 0 ; i < 2 * Impl::max_iterations ; i++) {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, right, left);
    }
  }
  Kokkos::finalize();
}
