export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export APEX_KOKKOS_TUNING_WINDOW=1
export APEX_KOKKOS_TUNING_POLICY=simulated_annealing
#export CUDA_VISIBLE_DEVICES=0,1,2

# Remove any cached results
rm -f apex_converged_tuning.yaml

# Baseline
/home/users/khuck/src/apex-kokkos-tuning/install/bin/apex_exec \
--apex:gtrace \
--apex:kokkos-fence \
--apex:cuda \
--apex:cuda-counters \
--apex:cuda-details \
/home/users/khuck/src/apex-kokkos-tuning/build/tests/mdrange_gemm \
--kokkos-device-id=1
mv trace_events.0.json.gz baseline.json.gz

# Tuning
/home/users/khuck/src/apex-kokkos-tuning/install/bin/apex_exec \
--apex:gtrace \
--apex:kokkos-fence \
--apex:cuda \
--apex:cuda-counters \
--apex:cuda-details \
--apex:kokkos-tuning \
/home/users/khuck/src/apex-kokkos-tuning/build/tests/mdrange_gemm \
--kokkos-tune-internals \
--kokkos-device-id=1
mv trace_events.0.json.gz tuning.json.gz

# Using cached results
/home/users/khuck/src/apex-kokkos-tuning/install/bin/apex_exec \
--apex:gtrace \
--apex:kokkos-fence \
--apex:cuda \
--apex:cuda-counters \
--apex:cuda-details \
--apex:kokkos-tuning \
/home/users/khuck/src/apex-kokkos-tuning/build/tests/mdrange_gemm \
--kokkos-tune-internals \
--kokkos-device-id=1
mv trace_events.0.json.gz cached.json.gz

