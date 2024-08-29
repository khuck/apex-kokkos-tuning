/**
 * 1D_stencil
 *
 * Complexity: medium
 * Tuning problem:
 *
 * Kokkos is executing a simple 1d stencil annealing (heat transfer) problem.
 *
 * This problem uses a Team policy for two instances, and the kernel
 * is the same for both instances. However, there are two Engine instances
 * to choose between: Static OpenMP and Dynamic OpenMP. This example will
 * tune the thread count and chunk size (vector length) too.
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
enum schedulers{StaticSchedule, DynamicSchedule};
static const std::string scheduleNames[] = {"static", "dynamic"};
namespace KTE = Kokkos::Tools::Experimental;
namespace KE = Kokkos::Experimental;

// helper function for matrix init
void initArray(Kokkos::View<double *, Kokkos::HostSpace>& ar, size_t d1) {
    for(size_t i=0; i<d1; i++){
        ar(i)=(rand() % (upperBound - lowerBound + 1)) + lowerBound;
    }
}

// Helper function to generate tile sizes
std::vector<int64_t> factorsOf(const int &size){
    std::vector<int64_t> factors;
    for(int i=1; i<size; i++){
        if(size % i == 0){
            factors.push_back(i);
        }
    }
    return factors;
}

// Helper function to generate thread counts
std::vector<int64_t> makeRange(const int &size){
    std::vector<int64_t> range;
    for(int i=2; i<=size; i+=2){
        range.push_back(i);
    }
    return range;
}

// helper function for human output
void reportOptions(std::vector<int64_t>& candidates,
    std::string name, size_t size) {
    std::cout<<"Chunk size options for " << name << "="<<size<<std::endl;
    for(auto &i : candidates){ std::cout<<i<<", "; }
    std::cout<<std::endl;
}

// helper function for declaring output tiling variables
size_t declareOutputTileSize(std::string name, std::string varname, size_t limit) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<int64_t> candidates = factorsOf(limit);
    reportOptions(candidates, name, limit);
    // create our variable object
    KTE::VariableInfo out_info;
    // set the variable details
    out_info.type = KTE::ValueType::kokkos_value_int64;
    out_info.category = KTE::StatisticalCategory::kokkos_value_categorical;
    out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    out_info.candidates = KTE::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = KTE::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

// helper function for declaring input size variables
size_t declareInputViewSize(std::string varname, int64_t size) {
    size_t in_value_id;
    // create a 'vector' of value(s)
    std::vector<int64_t> candidates = {size};
    // create our variable object
    KTE::VariableInfo in_info;
    // set the variable details
    in_info.type = KTE::ValueType::kokkos_value_int64;
    in_info.category = KTE::StatisticalCategory::kokkos_value_categorical;
    in_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    in_info.candidates = KTE::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    in_value_id = KTE::declare_input_type(varname,in_info);
    // return the id
    return in_value_id;
}

// helper function for declaring scheduler variable
size_t declareOutputSchedules(std::string varname) {
    // create a vector of potential values
    std::vector<int64_t> candidates_schedule = {StaticSchedule,DynamicSchedule};
    // create our variable object
    KTE::VariableInfo schedule_out_info;
    // set the variable details
    schedule_out_info.type = KTE::ValueType::kokkos_value_int64;
    schedule_out_info.category = KTE::StatisticalCategory::kokkos_value_categorical;
    schedule_out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    schedule_out_info.candidates = KTE::make_candidate_set(candidates_schedule.size(),candidates_schedule.data());
    // declare the variable
    size_t schedule_out_value_id = KTE::declare_output_type(varname,schedule_out_info);
    // return the id
    return schedule_out_value_id;
}

// helper function for declaring output tread count variable
size_t declareOutputThreadCount(std::string varname, size_t limit) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<int64_t> candidates = makeRange(limit);
    // create our variable object
    KTE::VariableInfo out_info;
    // set the variable details
    out_info.type = KTE::ValueType::kokkos_value_int64;
    out_info.category = KTE::StatisticalCategory::kokkos_value_categorical;
    out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    out_info.candidates = KTE::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = KTE::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

int main(int argc, char *argv[]) {
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

        // Context variable setup - needed to generate a unique context hash for tuning.
        // Declare the input variables and store the variable IDs
        size_t id[5];
        id[0] = 1; // default input for the region name (i.e. "openmp dynamic heat_transfer")
        id[1] = 2; // default input for the region type ("parallel_for")
        id[2] = declareInputViewSize("array_size", length);
        // create an input vector of variables with name, loop type, and array size.
        std::vector<KTE::VariableValue> input_vector{
            KTE::make_variable_value(id[0], "region name"),
            KTE::make_variable_value(id[1], "parallel_for"),
            KTE::make_variable_value(id[2], int64_t(length))
        };
        // Declare the ouptut variables and store the variable IDs
        size_t out_value_id[3];
        out_value_id[0] = declareOutputTileSize("length", "chunk_out", length);
        out_value_id[1] = declareOutputSchedules("schedule_out");
        int64_t max_threads = std::min(std::thread::hardware_concurrency(),
                (unsigned int)(Kokkos::OpenMP::concurrency()));
        out_value_id[2] = declareOutputThreadCount("thread_count", max_threads);
        //The second argument to make_varaible_value is a default value
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_value_id[0], int64_t(length/max_threads)),
            KTE::make_variable_value(out_value_id[1], int64_t(StaticSchedule)),
            KTE::make_variable_value(out_value_id[2], int64_t(max_threads))
        };



        /* We iterate so that we have enough samples to explore the search space.
         * In a real application, this kernel would get called multiple times over
         * the course of a simulation, and would eventually(?) converge. */
        for (int i = 0 ; i < Impl::max_iterations ; i++) {
            // request a context id
            size_t context = KTE::get_new_context_id();
            // start the context
            KTE::begin_context(context);
            // set the input values for the context
            KTE::set_input_values(context, input_vector.size(), input_vector.data());
            // request new output values for the context
            KTE::request_output_values(context, answer_vector.size(), answer_vector.data());
            // get the chunk size
            int chunk{static_cast<int>(answer_vector[0].value.int_value)};
            // get our schedule and thread count
            int scheduleType = answer_vector[1].value.int_value;
            // there's probably a better way to set the thread count?
            int num_threads = answer_vector[2].value.int_value;
            int league_size{1};

            // no tuning?
            if (!tuning) {
                Kokkos::parallel_for("openmp static heat_transfer",
                    Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Static>, Kokkos::OpenMP>(
                        league_size, Kokkos::AUTO, Kokkos::AUTO), kernel);
            } else if (scheduleType == StaticSchedule) {
                Kokkos::parallel_for("openmp dynamic heat_transfer",
                    Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Static>, Kokkos::OpenMP>(
                        league_size, num_threads, chunk), kernel);
            } else { // Dynamic schedule
                Kokkos::parallel_for("openmp dynamic heat_transfer",
                    Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::OpenMP>(
                        league_size, num_threads, chunk), kernel);
            }
            // end the context
            KTE::end_context(context);

            /* Swap the views */
            auto& tmp = source;
            source = dest;
            dest = tmp;
        }
    }
    Kokkos::finalize();
}
