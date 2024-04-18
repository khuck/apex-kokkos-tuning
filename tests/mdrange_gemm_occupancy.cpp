/**
 * idk_just_multiply_matrices
 *
 * Complexity: high
 *
 * Tuning problem:
 *
 * This is a *nested* tuning problem, with some complexity in the
 * inner tuning problems.
 *
 * This simulates a user who doesn't know Kokkos very well,
 * telling Kokkos to decide whether to do a matmul using
 * a TeamPolicy or an MDRangePolicy, they express no preference.
 *
 * If you pick an MDRangePolicy, that involves tuning a tile size,
 * as referenced in the "deep_copy" benchmark.
 *
 * If you pick a TeamPolicy, that involves tuning a "team_size"
 * and "vector_length," constructs that shape the amount of
 * parallelism in different levels of Kokkos.
 *
 * The "fastest_of" construct exposes a categorical choice among
 * implementations. Note that the tuning interface doesn't really
 * tell you that you're in a nested context, you'll just see
 *
 * begin_context(fastest_of_context_id)
 * request_values(which_implementation_should_i_use)
 * [suppose you say "TeamPolicy"]
 * begin_context(team_policy_tuner_id)
 * request_values(team_size, vector_length)
 * end_context(team_policy_tuner_id)
 * end_context(fastest_of_context_id)
 *
 * This is an extremely difficult problem
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
  constexpr const int data_size = 1000;
  using view_type =
      Kokkos::View<float **, Kokkos::DefaultExecutionSpace::memory_space>;

  tuned_kernel(
      argc, argv,
      [&](const int total_iters) {
        view_type left("left_inp", data_size, data_size);
        view_type right("right_inp", data_size, data_size);
        view_type output("output", data_size, data_size);
        return std::make_tuple(left, right, output);
      },
            [&](const int data_size, const int total_iters,
                view_type left, view_type right, view_type output) {
              Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                    Kokkos::Rank<2>> p(
                  {0, 0}, {data_size, data_size});
              auto const p_occ = Kokkos::Experimental::prefer(
                p, Kokkos::Experimental::DesiredOccupancy{Kokkos::AUTO});
              Kokkos::parallel_for("mdrange_gemm", p_occ,
                  KOKKOS_LAMBDA(const int x, const int y) {
                    for (int z = 0; z < data_size; ++z) {
                      output(x, y) += left(x, z) * right(z, y);
                    }
                  });
            });
}
