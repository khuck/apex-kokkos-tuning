#include <tuning_playground.hpp>
#include <omp.h>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
#include <unistd.h>
#include <ctime>
#include <random>

const int M=64;
const int N=64;
const int P=64;

using matrix2d = Kokkos::View<int **, Kokkos::OpenMP::memory_space>;
int lb=100, ub=999;

std::vector<int64_t> factorsOf(const int &size){

    std::vector<int64_t> factors;
    for(int i=1; i<size; i++){
        if(size % i == 0){
            factors.push_back(i);
        }
    }

    return factors;
}

void reportOptions(std::vector<int64_t>& candidates,
    std::string name, size_t size) {
    std::cout<<"Tiling options for " << name << "="<<size<<std::endl;
    for(auto &i : candidates){ std::cout<<i<<", "; }
    std::cout<<std::endl;
}

int main(int argc, char *argv[]){

  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout, false);
    srand(time(0));

    /* Declare/Init re,ar1,ar2 */
    matrix2d ar1("array1",M,N), ar2("array2",N,P), re("Result",M,P);

    /* input array 1 */
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
                ar1(i,j)=(rand() % (ub - lb + 1)) + lb;
        }
    }

    /* input array 2 */
    for(int j=0; j<N; j++){
        for(int k=0; k<P; k++){
                ar2(j,k)=(rand() % (ub - lb + 1)) + lb;
        }
    }

    /* output array 2 */
    for(int i=0; i<M; i++){
        for(int k=0; k<P; k++){
                re(i,k)=0;
        }
    }

    //Tuning tile size - setup

    size_t ti_out_value_id, tj_out_value_id, tk_out_value_id;
    std::vector<int64_t> candidates_ti = factorsOf(M);
    std::vector<int64_t> candidates_tj = factorsOf(N);
    std::vector<int64_t> candidates_tk = factorsOf(P);

    reportOptions(candidates_ti, "M", M);
    reportOptions(candidates_tj, "N", N);
    reportOptions(candidates_tk, "P", P);

    //Output variables - ti,tj,tk
    Kokkos::Tools::Experimental::VariableInfo ti_out_info, tj_out_info, tk_out_info;

    //Semantics of output: ti,tj,tk
    ti_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
    tj_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;;
    tk_out_info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;;

    ti_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
    tj_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;
    tk_out_info.category = Kokkos::Tools::Experimental::StatisticalCategory::kokkos_value_categorical;

    ti_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    tj_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
    tk_out_info.valueQuantity = Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;

    ti_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_ti.size(),candidates_ti.data());
    tj_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_tj.size(),candidates_tj.data());
    tk_out_info.candidates = Kokkos::Tools::Experimental::make_candidate_set(candidates_tk.size(),candidates_tk.data());

    //Declare Output Type

    ti_out_value_id = Kokkos::Tools::Experimental::declare_output_type("ti_out",ti_out_info);
    tj_out_value_id = Kokkos::Tools::Experimental::declare_output_type("tj_out",tj_out_info);
    tk_out_value_id = Kokkos::Tools::Experimental::declare_output_type("tk_out",tk_out_info);

    //Tuning tile size - end setup

    for (int i = 0 ; i < Impl::max_iterations ; i++) {
        //Tuning sile size
        //Initial values for the tile size ti,tj,tk
        ////int64_t ti_inp = iter%M, tj_inp = iter%N, tk_inp = iter%P;

        //The second argument to make_varaible_value might be a default value
        std::vector<Kokkos::Tools::Experimental::VariableValue> answer_vector{
            Kokkos::Tools::Experimental::make_variable_value(ti_out_value_id, int64_t(1)),
            Kokkos::Tools::Experimental::make_variable_value(tj_out_value_id, int64_t(1)),
            Kokkos::Tools::Experimental::make_variable_value(tk_out_value_id, int64_t(1))
        };

        size_t context = Kokkos::Tools::Experimental::get_new_context_id();
        Kokkos::Tools::Experimental::begin_context(context);
        ////Kokkos::Tools::Experimental::set_input_values(context, 3, feature_vector.data());
        Kokkos::Tools::Experimental::request_output_values(context, 3, answer_vector.data());

        int ti,tj,tk;
        ti = answer_vector[0].value.int_value;
        tj = answer_vector[1].value.int_value;
        tk = answer_vector[2].value.int_value;

        //End tuning tile size

        //Start a timer?
        //Kokkos::Timer timer;
        //Iteration Range
        Kokkos::MDRangePolicy<Kokkos::OpenMP,Kokkos::Rank<3>> policy({0,0,0},{M,N,P},{ti,tj,tk});
        //Iteration indices (i,j,k) here are mapped to cores and the cores executes the computational body for the given indicies.
        Kokkos::parallel_for(
            "mm2D", policy, KOKKOS_LAMBDA(int i, int j, int k){
                re(i,j) += ar1(i,j) * ar2(j,k);
            }
        );
        Kokkos::Tools::Experimental::end_context(context);
    }
  }
  Kokkos::finalize();
}