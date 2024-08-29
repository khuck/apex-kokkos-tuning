include_directories(${PROJECT_SOURCE_DIR}/tests)
add_custom_target(tuning.tests)

# Our set of programs
set(tuning_programs
    1d_stencil
    1d_stencil_chunk
    1d_stencil_team
    1d_stencil_team_auto
    2d_stencil
    deep_copy_2
    deep_copy_3
    deep_copy_4
    deep_copy_5
    deep_copy_6
    idk_jmm
    mm2d_tiling
    mdrange_gemm
    mdrange_gemm_occupancy
    occupancy
    )

# Our set of tuning methods to test
set(tuning_policies
    exhaustive
    genetic_search
    random
    simulated_annealing)

include(ProcessorCount)
ProcessorCount(NPROC)
if(${NPROC} EQUAL 0)
    set(NPROC 8)
else()
    if(${NPROC} LESS_EQUAL 8)
        set(NPROC 4)
    else()
        set(NPROC 8)
    endif()
endif()

foreach(tuning_prog ${tuning_programs})
    set(sources ${tuning_prog}.cpp)
    source_group("Source Files" FILES ${sources})
    message(INFO " Adding test program: ${tuning_prog}")
    add_executable(${tuning_prog} ${sources})
    target_link_libraries(${tuning_prog} kokkos)
    add_dependencies (${tuning_prog} apex)
    add_dependencies (tuning.tests ${tuning_prog})
    # This is needed to make sure local symbols are exported and we can dladdr them
    set_property(TARGET ${tuning_prog} PROPERTY ENABLE_EXPORTS ON)

    # Do one test without any tuning
    add_test (NAME test_${tuning_prog}_no_tuning
        COMMAND ${CMAKE_BINARY_DIR}/apex/src/scripts/apex_exec --apex:kokkos-fence ${CMAKE_BINARY_DIR}/tests/${tuning_prog})
    set_tests_properties(test_${tuning_prog}_no_tuning PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=${NPROC};OMP_PROC_BIND=spread;OMP_PLACES=threads")

    foreach(tuning_policy ${tuning_policies})
        # Test the tuning
        add_test (NAME test_${tuning_prog}_${tuning_policy}
            COMMAND ${CMAKE_BINARY_DIR}/apex/src/scripts/apex_exec --apex:kokkos-tuning ${CMAKE_BINARY_DIR}/tests/${tuning_prog} --kokkos-tune-internals)
        # Test the cached convergence usage
        add_test (NAME test_${tuning_prog}_${tuning_policy}_cached
            COMMAND ${CMAKE_BINARY_DIR}/apex/src/scripts/apex_exec --apex:kokkos-tuning ${CMAKE_BINARY_DIR}/tests/${tuning_prog} --kokkos-tune-internals)
        # Add a cleanup step to remove the convergence cache file
        add_test(NAME test_${tuning_prog}_${tuning_policy}_cleanup
            COMMAND ${CMAKE_COMMAND} -E
                rm -f "${CMAKE_CURRENT_BINARY_DIR}/apex_converged_tuning.yaml")
        set_tests_properties(test_${tuning_prog}_${tuning_policy}_cleanup
            PROPERTIES FIXTURES_CLEANUP test_${tuning_prog}_${tuning_policy}_teardown)
        if(${tuning_policy} STREQUAL "exhaustive" OR
           ${tuning_policy} STREQUAL "random")
            if(${tuning_prog} STREQUAL "1d_stencil")
                set(window "5")
            elseif(${tuning_prog} STREQUAL "2d_stencil")
                set(window "2")
            else()
                set(window "1")
            endif()
        elseif(${tuning_policy} STREQUAL "simulated_annealing" AND
               ${tuning_prog} STREQUAL "idk_jmm")
            set(window "1")
        else()
            set(window "2")
        endif()
        # the 1d_stencil and 2d_stencil can have bigger windows
        set_tests_properties(test_${tuning_prog}_${tuning_policy} PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=${NPROC};OMP_PROC_BIND=spread;OMP_PLACES=threads;APEX_KOKKOS_TUNING_WINDOW=${window};APEX_KOKKOS_TUNING_POLICY=${tuning_policy}"
            PASS_REGULAR_EXPRESSION "Tuning has converged"
            )
        set_tests_properties(test_${tuning_prog}_${tuning_policy}_cached PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=${NPROC};OMP_PROC_BIND=spread;OMP_PLACES=threads;APEX_KOKKOS_TUNING_WINDOW=${window};APEX_KOKKOS_TUNING_POLICY=${tuning_policy}"
            PASS_REGULAR_EXPRESSION "Reading cache of Kokkos tuning results;Tuning has converged;New best"
            FIXTURES_REQUIRED test_${tuning_prog}_${tuning_policy}_teardown)
    endforeach()
endforeach()

set_tests_properties(test_deep_copy_4_exhaustive test_deep_copy_5_exhaustive test_deep_copy_6_exhaustive test_mm2d_tiling_exhaustive PROPERTIES WILL_FAIL TRUE)
add_custom_command(TARGET tuning.tests POST_BUILD COMMAND ctest -R test --output-on-failure --timeout 180)

