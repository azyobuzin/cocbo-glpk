cmake_minimum_required(VERSION 3.10)
project(COCBO CXX)

find_package(fmt REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MLPACK REQUIRED mlpack>=3.0.0)

# mlpack uses Boost
find_package(Boost REQUIRED COMPONENTS program_options serialization unit_test_framework)

# mlpack uses OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(cocbo main.cpp cocbo.cpp)
target_link_libraries(cocbo fmt::fmt glpk ${Boost_LIBRARIES} ${MLPACK_LINK_LIBRARIES})
