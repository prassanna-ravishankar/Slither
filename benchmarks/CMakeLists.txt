# Benchmarks for Slither Random Forest

# Simple performance benchmark
add_executable(benchmark_classification benchmark_classification.cpp)
target_link_libraries(benchmark_classification 
    PRIVATE 
        Slither::Slither
)

# Add more benchmarks as needed
# add_executable(benchmark_regression benchmark_regression.cpp)
# target_link_libraries(benchmark_regression PRIVATE Slither::Slither)