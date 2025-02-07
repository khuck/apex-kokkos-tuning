/**
 * meta-smoother
 */

#include <Kokkos_Core.hpp>
#include <unordered_map>
#include <iostream>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <cstdlib>
#include <random>
#include <tuple>
#include "tuning_playground.hpp"
#include <chrono>
#include <thread>

namespace KTE = Kokkos::Tools::Experimental;
namespace KE = Kokkos::Experimental;

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

// Helper function to generate a linear series
template<typename T>
std::vector<T> makeRange(const T& min, const T& max, const T& step){
    std::vector<T> range;
    for(T i=min; i<=max; i+=step){
        range.push_back(i);
    }
    return range;
}

// helper function for human output
template<typename T>
void reportOptions(const std::vector<T>& candidates,
    std::string name) {
    std::string tmpstr{"Options for "};
    tmpstr += name;
    tmpstr += " [";
    for(auto &i : candidates){ tmpstr += std::to_string(i) + ",";}
    tmpstr[tmpstr.size()-1] = ']';
    std::cout << tmpstr << std::endl;
}

// helper function for human output
void reportOptions(const std::string& name, const double& lower,
    const double& upper, const bool& openLower, const bool& openUpper) {
    std::string tmpstr{"Options for "};
    tmpstr += name;
    tmpstr += (openLower ? "(" : "[");
    tmpstr += std::to_string(lower) + "," + std::to_string(upper);
    tmpstr += (openUpper ? ")" : "]");
    std::cout << tmpstr << std::endl;
}

// helper function for declaring output tiling variables
size_t declareOutputTileSize(std::string name, std::string varname, size_t limit) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<int64_t> candidates = factorsOf(limit);
    reportOptions(candidates, name);
    // create our variable object
    KTE::VariableInfo out_info;
    // set the variable details
    out_info.type = KTE::ValueType::kokkos_value_int64;
    out_info.category = KTE::StatisticalCategory::kokkos_value_ordinal;
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
    in_info.category = KTE::StatisticalCategory::kokkos_value_ordinal;
    in_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    in_info.candidates = KTE::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    in_value_id = KTE::declare_input_type(varname,in_info);
    // return the id
    return in_value_id;
}

/*
// helper function for declaring scheduler variable
size_t declareOutputSchedules(std::string varname) {
    // create a vector of potential values
    std::vector<int64_t> candidates_schedule = {StaticSchedule,DynamicSchedule};
    // create our variable object
    KTE::VariableInfo schedule_out_info;
    // set the variable details
    schedule_out_info.type = KTE::ValueType::kokkos_value_int64;
    schedule_out_info.category = KTE::StatisticalCategory::kokkos_value_ordinal;
    schedule_out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    schedule_out_info.candidates = KTE::make_candidate_set(candidates_schedule.size(),candidates_schedule.data());
    // declare the variable
    size_t schedule_out_value_id = KTE::declare_output_type(varname,schedule_out_info);
    // return the id
    return schedule_out_value_id;
}
*/

// helper function for declaring range of int64_t values
size_t declareOutputRangeInt64(const std::string varname,
        const int64_t& lower, const int64_t& upper, const int64_t& step) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<int64_t> candidates = makeRange<int64_t>(lower, upper, step);
    reportOptions(candidates, varname);
    // create our variable object
    KTE::VariableInfo out_info;
    // set the variable details
    out_info.type = KTE::ValueType::kokkos_value_int64;
    out_info.category = KTE::StatisticalCategory::kokkos_value_ordinal;
    out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    out_info.candidates = KTE::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = KTE::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

// helper function for declaring range of double values
size_t declareOutputRangeDouble(const std::string varname,
        const double& lower, const double& upper, const double& step) {
    size_t out_value_id;
    // create a vector of potential values
    std::vector<double> candidates = makeRange<double>(lower, upper, step);
    reportOptions(candidates, varname);
    // create our variable object
    KTE::VariableInfo out_info;
    // set the variable details
    out_info.type = KTE::ValueType::kokkos_value_double;
    out_info.category = KTE::StatisticalCategory::kokkos_value_ordinal;
    out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_set;
    out_info.candidates = KTE::make_candidate_set(candidates.size(),candidates.data());
    // declare the variable
    out_value_id = KTE::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

// helper function for declaring range of double values
size_t declareOutputContinuous(const std::string varname,
        const double& lower, const double& upper, const double& step,
        bool openLower, bool openUpper) {
    size_t out_value_id;
    reportOptions(varname, lower, upper, openLower, openUpper);
    // create our variable object
    KTE::VariableInfo out_info;
    // set the variable details
    out_info.type = KTE::ValueType::kokkos_value_double;
    out_info.category = KTE::StatisticalCategory::kokkos_value_interval;
    out_info.valueQuantity = KTE::CandidateValueType::kokkos_value_range;
    out_info.candidates = KTE::make_candidate_range(lower, upper, step, openLower, openUpper);
    // declare the variable
    out_value_id = KTE::declare_output_type(varname,out_info);
    // return the id
    return out_value_id;
}

// helper function for declaring generic range of values
template<typename T>
size_t declareOutputRange(const std::string varname,
        const T lower, const T upper, const T step) {
    if(typeid(T) == typeid(int64_t)) {
        return declareOutputRangeInt64(varname, lower, upper, step);
    } else if(typeid(T) == typeid(double)) {
        return declareOutputRangeDouble(varname, lower, upper, step);
    } else {
        assert(false);
    }
}
namespace metasmoother {
    std::vector<KTE::VariableValue> makeChebychevVariables() {
        // output variable ids
        size_t out_variables[3];
        out_variables[0] = declareOutputRange<int64_t>("Chebyshev Degree", 1, 6, 1);
        out_variables[1] = declareOutputContinuous("Eigenvalue Ratio", 10.0, 50.0, 0.1, false, false);
        out_variables[2] = declareOutputRange<int64_t>("Maximum Chebychev Iterations", 5, 100, 1);
        //The second argument to make_varaible_value might be a default value
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_variables[0], int64_t(3)),
            KTE::make_variable_value(out_variables[1], double(25.0)),
            KTE::make_variable_value(out_variables[2], int64_t(50))
        };
        return answer_vector;
    }

    void doChebyshev(void) {
        Kokkos::Profiling::ScopedRegion region("Chebyshev");
        //std::cout << " Doing Chebyshev..." << std::endl;
        size_t context{KTE::get_new_context_id()};
        KTE::begin_context(context);

        // set the input values for the context
        //KTE::set_input_values(context, input_vector.size(), input_vector.data());

        // set the output values for the context
        static std::vector<KTE::VariableValue> answer_vector{makeChebychevVariables()};

        // request new output values for the context
        // get the settings...
        KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

        // try to converge on 5, 75, 15
        size_t delay = 1 + (std::abs(5 - answer_vector[0].value.int_value) * 10) +
                           (std::abs(15.0 - answer_vector[1].value.double_value) * 10) +
                           (std::abs(75 - answer_vector[2].value.int_value) * 10);
        // call the real solver
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        // end the context
        KTE::end_context(context);
    }

    std::vector<KTE::VariableValue> makeMultiThreadedGaussSeidelVariables() {
        // output variable ids
        size_t out_variables[3];
        out_variables[0] = declareOutputRange<int64_t>("Number of Sweeps", 1, 2, 1);
        out_variables[1] = declareOutputContinuous("Damping Factor", 0.8, 1.2, 0.01, false, false);
        //The second argument to make_varaible_value might be a default value
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_variables[0], int64_t(2)),
            KTE::make_variable_value(out_variables[1], double(1.0)),
        };
        return answer_vector;
    }

    void MultiThreadedGaussSeidel(void) {
        Kokkos::Profiling::ScopedRegion region("Multi-threaded Gauss-Seidel");
        //std::cout << " Doing Multi-threaded Gauss-Seidel..." << std::endl;
        size_t context{KTE::get_new_context_id()};
        KTE::begin_context(context);

        // set the input values for the context
        //KTE::set_input_values(context, input_vector.size(), input_vector.data());

        // set the output values for the context
        static std::vector<KTE::VariableValue> answer_vector{makeMultiThreadedGaussSeidelVariables()};

        // request new output values for the context
        // get the settings...
        KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

        // try to converge on 1, 0.9
        size_t delay = 1 + (std::abs(1 - answer_vector[0].value.int_value) * 100) +
                           (std::abs(0.9 - answer_vector[1].value.double_value) * 100);
        // call the real solver
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        // end the context
        KTE::end_context(context);
    }

    std::vector<KTE::VariableValue> makeTwoStageGaussSeidelVariables() {
        // output variable ids
        size_t out_variables[3];
        out_variables[0] = declareOutputRange<int64_t>("Number of Sweeps", 1, 2, 1);
        out_variables[1] = declareOutputContinuous("Inner Damping Factor", 0.8, 1.2, 0.01, false, false);
        //The second argument to make_varaible_value might be a default value
        std::vector<KTE::VariableValue> answer_vector{
            KTE::make_variable_value(out_variables[0], int64_t(2)),
            KTE::make_variable_value(out_variables[1], double(1.0)),
        };
        return answer_vector;
    }

    void TwoStageGaussSeidel(void) {
        Kokkos::Profiling::ScopedRegion region("Two-Stage Gauss-Seidel");
        //std::cout << " Doing Two-Stage Gauss-Seidel..." << std::endl;
        size_t context{KTE::get_new_context_id()};
        KTE::begin_context(context);

        // set the input values for the context
        //KTE::set_input_values(context, input_vector.size(), input_vector.data());

        // set the output values for the context
        static std::vector<KTE::VariableValue> answer_vector{makeTwoStageGaussSeidelVariables()};

        // request new output values for the context
        // get the settings...
        KTE::request_output_values(context, answer_vector.size(), answer_vector.data());

        // call the real solver
        // try to converge on 2, 1.1
        size_t delay = 1 + (std::abs(2 - answer_vector[0].value.int_value) * 100) +
                           (std::abs(1.1 - answer_vector[1].value.double_value) * 100);
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        // end the context
        KTE::end_context(context);
    }
};

int main(int argc, char *argv[]) {

    Kokkos::initialize(argc, argv);
    {
        Kokkos::print_configuration(std::cout, false);
        Kokkos::Profiling::ScopedRegion region("meta smoother search loop");
        for (int i = 0 ; i < 1000 ; i++) {
            fastest_of("meta-smoother", 3,
                [&]() { metasmoother::doChebyshev(); },
                [&]() { metasmoother::MultiThreadedGaussSeidel(); },
                [&]() { metasmoother::TwoStageGaussSeidel(); }
            );
        }
    }
    Kokkos::finalize();
}
