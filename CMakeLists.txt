cmake_minimum_required(VERSION 3.8)

### Hunter snapshot that will be used ###
include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.23.33.tar.gz"
        SHA1 "acbf4b9b77b5a0160adc1da1cdda35f64b14196a"
        LOCAL
)
#include("cmake/config.cmake")
set(HUNTER_KEEP_PACKAGE_SOURCES NO)

project(randomSlither)


#######################################
# Dependencies
#######################################
# Required :  Boost - Move python interface to pybind11
#set(BUILD_SHARED_LIBS ON)
add_definitions(-DBOOST_ALL_NO_LIB=1)
set(Boost_USE_STATIC_LIBS OFF)
set(BUILD_SHARED_LIBS)
hunter_add_package(Boost COMPONENTS program_options serialization)
find_package(Boost CONFIG REQUIRED program_options serialization)
set_target_properties(Boost::serialization PROPERTIES POSITION_INDEPENDENT_CODE TRUE) # Adding -fPIC to boost

# Required : Opencv
find_package( OpenCV CONFIG REQUIRED)


# For the library
set(SOURCE_FILES
        source/Classification.cpp
        source/CommandLineParser.cpp
        source/CumulativeNormalDistribution.cpp
        source/DataPointCollection.cpp
        #demo/source/DensityEstimation.cpp
        source/dibCodec.cpp
        source/FeatureResponseFunctions.cpp
        source/FloydWarshall.cpp
        source/Graphics.cpp
        source/Platform.cpp
        source/PlotCanvas.cpp
        #demo/source/Regression.cpp
        #demo/source/SemiSupervisedClassification.cpp
        source/StatisticsAggregators.cpp)


#######################################
# Build the library
#######################################
add_library(Slither SHARED ${SOURCE_FILES})
# Add related includes
target_include_directories(Slither PUBLIC lib)
target_include_directories(Slither PUBLIC source)
target_include_directories(Slither PUBLIC Boost::serialization)
# Link related libs
target_link_libraries(Slither PUBLIC ${OpenCV_LIBS})
target_link_libraries(Slither PUBLIC Boost::serialization)
set_target_properties(Slither PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )
# To allow internal shared libs to work and pybind11 to be okay
#set_target_properties(Slither PROPERTIES POSITION_INDEPENDENT_CODE TRUE)



#######################################
# Build the Standalone executable
#######################################
add_executable(cppSlither source/main.cpp)
# Linking and including
target_link_libraries(cppSlither PUBLIC Slither)
target_link_libraries(cppSlither PUBLIC Boost::program_options)

set_target_properties(cppSlither PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        )


#######################################
# Build the Python Wrapper
#######################################
# First --> Python requirements
hunter_add_package(pybind11)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pySlither pyWrapper/wrapper.cpp)
target_link_libraries(pySlither PUBLIC Slither)
#set_target_properties(pySlither PROPERTIES POSITION_INDEPENDENT_CODE TRUE)


