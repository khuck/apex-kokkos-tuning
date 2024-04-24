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
  constexpr const int data_size = 1024;
  using view_type =
      Kokkos::View<float **, Kokkos::DefaultExecutionSpace::memory_space>;

  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout, false);
    view_type left("left_inp", data_size, data_size);
    view_type right("right_inp", data_size, data_size);
    view_type output("output", data_size, data_size);

    for (int i = 0 ; i < Impl::max_iterations ; i++) {
        fastest_of(
            "bad_gemms", 2,
            [&]() {
              //std::cout << i << " Doing team gemm..." << std::endl;
              using team_policy =
                  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
              using team_member = team_policy::member_type;
              Kokkos::parallel_for(
                  "bad_team_gemm",
                  team_policy(data_size * data_size, Kokkos::AUTO,
                              Kokkos::AUTO),
                  KOKKOS_LAMBDA(const team_member &member) {
                    auto index = member.league_rank();
                    auto x = index % data_size;
                    auto y = index / data_size;
                    float sum = 0;
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(member, data_size),
                        [&](int &i, float &lsum) {
                          lsum += left(x, i) * right(i, y);
                        },
                        sum);
                    output(x, y) = sum;
                  });
            },
            [&]() {
              //std::cout << i << " Doing mdrange gemm..." << std::endl;
              Kokkos::parallel_for(
                  "bad_mdrange_gemm",
                  Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                        Kokkos::Rank<2>>(
                      {0, 0}, {data_size, data_size}),
                  KOKKOS_LAMBDA(const int x, const int y) {
                    for (int z = 0; z < data_size; ++z) {
                      output(x, y) += left(x, z) * right(z, y);
                    }
                  });
            });
    }
  }
  Kokkos::finalize();
}
