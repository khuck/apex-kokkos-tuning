include_directories(${PROJECT_SOURCE_DIR}/tests)
add_custom_target(tuning.tests)

set(tuning_programs
    occupancy
    mm2d_tiling
    idk_jmm
    deep_copy
    mdrange_gemm
    mdrange_gemm_occupancy)

foreach(tuning_prog ${tuning_programs})
    set(sources ${tuning_prog}.cpp)
    source_group("Source Files" FILES ${sources})
    message(INFO " Adding test program: ${tuning_prog}")
    add_executable(${tuning_prog} ${sources})
    target_link_libraries(${tuning_prog} kokkos)
    add_dependencies (${tuning_prog} apex)
    add_dependencies (tuning.tests ${tuning_prog})
    add_test (NAME test_${tuning_prog} COMMAND ${CMAKE_BINARY_DIR}/apex/src/scripts/apex_exec --apex:kokkos-tuning ${CMAKE_BINARY_DIR}/tests/${tuning_prog} --kokkos-tune-internals)
    set_tests_properties(test_${tuning_prog} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=8;OMP_PROC_BIND=spread;OMP_PLACES=threads;APEX_KOKKOS_TUNING_WINDOW=5;APEX_KOKKOS_TUNING_POLICY=simulated_annealing")
endforeach()

add_custom_command(TARGET tuning.tests POST_BUILD COMMAND ctest -R test --verbose --output-on-failure --timeout 100)

#add_executable(occupancy occupancy.cpp)
#target_link_libraries(occupancy kokkos)
#add_dependencies (tuning.tests occupancy)

#add_executable(mm2d_tiling mm2d_tiling.cpp)
#target_link_libraries(mm2d_tiling kokkos)

#add_executable(idk_jmm idk_jmm.cpp)
#target_link_libraries(idk_jmm kokkos)

#add_executable(deep_copy deep_copy.cpp)
#target_link_libraries(deep_copy kokkos)

#add_executable(mdrange_gemm mdrange_gemm.cpp)
#target_link_libraries(mdrange_gemm kokkos)

#add_executable(mdrange_gemm_occupancy mdrange_gemm_occupancy.cpp)
#target_link_libraries(mdrange_gemm_occupancy kokkos)

