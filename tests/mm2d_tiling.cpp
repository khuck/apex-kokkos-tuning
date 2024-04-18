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

int main(int argc, char *argv[]){

    tuned_kernel(
        argc, argv,
        [&](const int total_iters) {

            srand(time(0));

            /* Declare/Init re,ar1,ar2 */
            matrix2d ar1("array1",M,N), ar2("array2",N,P), re("Result",M,P);

            for(int i=0; i<M; i++){
                for(int j=0; j<N; j++){
                        ar1(i,j)=(rand() % (ub - lb + 1)) + lb;
                        //ar1(i,j)=5;
                }
            }

            for(int j=0; j<N; j++){
                for(int k=0; k<P; k++){
                        ar2(j,k)=(rand() % (ub - lb + 1)) + lb;
                        //ar2(j,k)=5;
                }
            }

            for(int i=0; i<M; i++){
                for(int k=0; k<P; k++){
                        re(i,k)=0;
                }
            }

            //Tuning tile size

            ////size_t ti_inp_value_id, tj_inp_value_id, tk_inp_value_id;
                size_t ti_out_value_id, tj_out_value_id, tk_out_value_id;
            /*
            std::vector<int64_t> candidates_ti{1,16,32,64,128,256,512},
                                candidates_tj{1,16,32,64,128,256,512},
                                candidates_tk{1,16,32,64,128,256,512};
            */

            std::vector<int64_t> candidates_ti = factorsOf(M);
            std::vector<int64_t> candidates_tj = factorsOf(N);
            std::vector<int64_t> candidates_tk = factorsOf(P);

            std::cout<<"Tiling options for M="<<M<<std::endl;
            for(auto &i : candidates_ti){ std::cout<<i<<", "; }
            std::cout<<std::endl;

            std::cout<<"Tiling options for N="<<N<<std::endl;
            for(auto &i : candidates_tj){ std::cout<<i<<", "; }
            std::cout<<std::endl;

            std::cout<<"Tiling options for P="<<P<<std::endl;
            for(auto &i : candidates_tk){ std::cout<<i<<", "; }
            std::cout<<std::endl;

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

            //End tuning tile size
            //Need to return views too
            ////return std::make_tuple(ti_inp_value_id,tj_inp_value_id,tk_inp_value_id,
                                    ////ti_out_value_id,tj_out_value_id,tk_out_value_id,
                                    ////re, ar1, ar2);
            return std::make_tuple(ti_out_value_id,tj_out_value_id,tk_out_value_id,
                                    re, ar1, ar2);
        },
        ////[&] (const int iter, const int total_iters,
                ////size_t ti_inp_value_id, size_t tj_inp_value_id, size_t tk_inp_value_id,
                ////size_t ti_out_value_id, size_t tj_out_value_id, size_t tk_out_value_id,
                ////matrix2d re, matrix2d ar1, matrix2d ar2){

        [&] (const int iter, const int total_iters,
                size_t ti_out_value_id, size_t tj_out_value_id, size_t tk_out_value_id,
                matrix2d re, matrix2d ar1, matrix2d ar2){

            //Tuning sile size
            //Initial values for the tile size ti,tj,tk
            ////int64_t ti_inp = iter%M, tj_inp = iter%N, tk_inp = iter%P;

            ////std::vector<Kokkos::Tools::Experimental::VariableValue> feature_vector{
                ////Kokkos::Tools::Experimental::make_variable_value(ti_inp_value_id, ti_inp),
                ////Kokkos::Tools::Experimental::make_variable_value(tj_inp_value_id, tj_inp),
                ////Kokkos::Tools::Experimental::make_variable_value(tk_inp_value_id, tk_inp)
            ////};

            //The second argument to make_varaible_value might be a default value
            std::vector<Kokkos::Tools::Experimental::VariableValue> answer_vector{
                Kokkos::Tools::Experimental::make_variable_value(ti_out_value_id, int64_t(1)),
                Kokkos::Tools::Experimental::make_variable_value(tj_out_value_id, int64_t(1)),
                Kokkos::Tools::Experimental::make_variable_value(tk_out_value_id, int64_t(1))
            };

#if 1
            size_t context = Kokkos::Tools::Experimental::get_new_context_id();
            Kokkos::Tools::Experimental::begin_context(context);
            ////Kokkos::Tools::Experimental::set_input_values(context, 3, feature_vector.data());
            Kokkos::Tools::Experimental::request_output_values(context, 3, answer_vector.data());
#endif

            int ti,tj,tk;
            ti = answer_vector[0].value.int_value;
            tj = answer_vector[1].value.int_value;
            tk = answer_vector[2].value.int_value;


            //End tuning tile size

            //Start a timer?
            //Kokkos::Timer timer;
            //Iteration Range
            Kokkos::MDRangePolicy<Kokkos::OpenMP,Kokkos::Rank<3>> policy({0,0,0},{M,N,P},{ti,tj,tk});
            auto const p_occ = Kokkos::Experimental::prefer(
                policy, Kokkos::Experimental::DesiredOccupancy{Kokkos::AUTO});
            //Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::OpenMP, Kokkos::Schedule<static|dynamic|...>>, policy({0,0,0},{M,N,P},{tile_M,tile_N,tile_P})

            //Iteration indices (i,j,k) here are mapped to cores and the cores executes the computational body for the given indicies.
            Kokkos::parallel_for(
                "mm2D", p_occ, KOKKOS_LAMBDA(int i, int j, int k){
                    re(i,j) += ar1(i,j) * ar2(j,k);
                }
            );

            /*Kokkos::fence();
            double time_elapsed = timer.seconds();
            cout<<"Execution Time: "<<time_elapsed*1e6<<" microseconds"<<endl;*/

#if 1
            Kokkos::Tools::Experimental::end_context(context);
#endif
        }
    );
}